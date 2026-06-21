import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon
import AudioCommon

/// Synthesis progress logs go to stderr so they don't corrupt a stdout-based
/// IPC channel (e.g. speech-studio sidecar's NDJSON protocol). Swift's stdio
/// `print()` is line-buffered and may flush at unexpected boundaries; routing
/// to stderr keeps the stdout pipe clean for JSON responses.
@inline(__always)
private func iclLog(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

// MARK: - ICL Voice Cloning

extension Qwen3TTSModel {

    /// Load a Qwen3-TTS model together with the SpeechTokenizerEncoder for ICL voice cloning.
    ///
    /// The encoder is used to convert reference audio into codec tokens. Weights come from
    /// the same safetensors file (encoder.* prefixes).
    public static func fromPretrainedWithEncoder(
        // Default to the 1.7B-bf16 (the production model Studio ships) — NOT the
        // decommissioned 0.6B int4. The int4 default previously masked the 1.7B
        // speaker-encoder bug (2048-dim fc + PyTorch conv layout), since the e2e
        // tests only ever exercised the 0.6B path.
        modelId: String = "aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-bf16",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> (Qwen3TTSModel, SpeechTokenizerEncoder) {
        let tts = try await Qwen3TTSModel.fromPretrained(
            modelId: modelId, cacheDir: cacheDir, offlineMode: offlineMode, progressHandler: progressHandler)

        let weightsDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Build encoder with same config as decoder
        let encoderConfig = tts.config.speechTokenizerDecoder
        let encoder = SpeechTokenizerEncoder(config: encoderConfig)
        try TTSWeightLoader.loadSpeechTokenizerEncoderWeights(into: encoder, from: weightsDir)

        return (tts, encoder)
    }

    /// Synthesize speech using In-Context Learning (ICL) voice cloning.
    ///
    /// Unlike `synthesizeWithVoiceClone()` (x-vector only), ICL encodes the reference audio
    /// into codec tokens and prepends them with their transcript into the autoregressive
    /// context. This produces correct EOS and higher voice fidelity.
    ///
    /// Because the model is conditioned on `[reference_text + target_text]` and trained to
    /// produce codec for the entire text sequence, the raw decoded waveform contains a
    /// regeneration of the reference audio followed by the target. By default this method
    /// returns only the target portion; pass `trimReference: false` to receive the full
    /// waveform (useful for debugging the model's reference reproduction).
    ///
    /// - Parameters:
    ///   - text: Target text to synthesize
    ///   - referenceAudio: Raw PCM samples of the reference recording
    ///   - referenceSampleRate: Sample rate of referenceAudio (resampled to 24kHz internally)
    ///   - referenceText: Exact transcript of the reference recording
    ///   - language: Language hint (e.g. "english", "german")
    ///   - sampling: Sampling config
    ///   - codecEncoder: SpeechTokenizerEncoder from fromPretrainedWithEncoder()
    ///   - trimReference: When `true` (default), strip the reference-text portion from the
    ///                    start of the output, returning only the target audio. The trim
    ///                    point is estimated from the reference duration (codec runs at
    ///                    12.5 fps, 1920 samples per frame at 24 kHz). Set `false` to
    ///                    receive the raw decoded waveform.
    public func synthesizeWithVoiceCloneICL(
        text: String,
        referenceAudio: [Float],
        referenceSampleRate: Int = 24000,
        referenceText: String,
        language: String = "auto",
        sampling: SamplingConfig = .default,
        codecEncoder: SpeechTokenizerEncoder,
        trimReference: Bool = true
    ) -> [Float] {
        guard let tokenizer = tokenizer else {
            fatalError("Tokenizer not loaded")
        }
        // "auto" (matches the QwenLM reference + mlx-audio default) skips the
        // language-id token and switches the codec prefix to the codec_nothink
        // branch. Any other value must resolve to a known language id.
        let langId: Int?
        let normalized = language.lowercased()
        if normalized == "auto" || normalized.isEmpty {
            langId = nil
        } else if let id = CodecTokens.languageId(for: language) {
            langId = id
        } else {
            iclLog("Warning: Unknown language '\(language)', falling back to auto")
            return synthesizeWithVoiceCloneICL(
                text: text, referenceAudio: referenceAudio,
                referenceSampleRate: referenceSampleRate,
                referenceText: referenceText, language: "auto",
                sampling: sampling, codecEncoder: codecEncoder,
                trimReference: trimReference)
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Step 1+2: Encode reference audio → codec tokens [1, 16, T_ref] (cached per reference)
        let refCodes: MLXArray
        if let cached = referenceAudioCache.codecRefCodes(for: referenceAudio, sampleRate: referenceSampleRate) {
            refCodes = cached
            iclLog("  ICL: codec tokens cache hit (\(cached.dim(2)) frames)")
        } else {
            let audio24k = referenceSampleRate == 24000
                ? referenceAudio
                : AudioFileLoader.resample(referenceAudio, from: referenceSampleRate, to: 24000)
            let codes = codecEncoder.encode(samples: audio24k)
            eval(codes)
            referenceAudioCache.storeCodecRefCodes(codes, audio: referenceAudio, sampleRate: referenceSampleRate)
            refCodes = codes
            iclLog("  ICL: encoded \(audio24k.count) samples → \(codes.dim(2)) codec frames")
        }

        // Step 3: Extract speaker embedding (ICL still uses x-vector for speaker conditioning; cached)
        let speakerEmbed: MLXArray
        if let cached = referenceAudioCache.speakerEmbed(for: referenceAudio, sampleRate: referenceSampleRate) {
            speakerEmbed = cached
        } else {
            let mels = SpeakerMel.compute(audio: referenceAudio, sampleRate: referenceSampleRate)
            let embed = speakerEncoder(mels)  // [1, 1024]
            eval(embed)
            referenceAudioCache.storeSpeakerEmbed(embed, audio: referenceAudio, sampleRate: referenceSampleRate)
            speakerEmbed = embed
        }

        // Step 4: Build ICL prefill embeddings
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildICLPrefillEmbeddings(
            refCodes: refCodes,
            referenceText: referenceText,
            targetText: text,
            language: language,
            languageId: langId,
            speakerEmbed: speakerEmbed,
            tokenizer: tokenizer)

        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)
        let t1 = CFAbsoluteTimeGetCurrent()

        // Step 5: Autoregressive generation. Auto-bump repetition_penalty to
        // >= 1.5 ONLY when caller is sampling (T > 0). Under sampling the
        // bump matches mlx-audio's recipe and prevents codec-token repetition
        // artifacts. Under greedy the bump is a degeneration trap (argmax
        // flips on Metal logit jitter) — leave the caller's value as-is.
        var iclSampling = sampling
        if iclSampling.temperature > 0 && iclSampling.repetitionPenalty < 1.5 {
            iclSampling.repetitionPenalty = 1.5
        }
        // Cap maxTokens so under-EOS runaway outputs can't exhaust GPU memory.
        // ICL generates [ref reproduction + target], so the budget must cover
        // BOTH: ref-codec-frames (from the encoded reference) plus a per-target-
        // text allowance. 6 codec frames per text token is a loose upper bound
        // on natural speech rate (~12.5 fps / ~2 tok/s). Extra 1.5× safety
        // margin on the reference portion handles model speech-rate variance.
        let refCodecFrames = refCodes.dim(2)
        let targetTokenCount = tokenizer.encode(text).count
        let refBudget = refCodecFrames + refCodecFrames / 2  // 1.5x of ref frames
        let targetBudget = max(75, targetTokenCount * 6)
        let textDerivedCap = refBudget + targetBudget
        iclSampling.maxTokens = min(iclSampling.maxTokens, textDerivedCap)
        let (allCodebooks, numFrames) = generateWithCodePredictor(
            prefillEmbeds: prefillEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            sampling: iclSampling)

        eval(allCodebooks)
        let t2 = CFAbsoluteTimeGetCurrent()

        guard numFrames > 0 else {
            iclLog("Warning: ICL generation produced no tokens")
            return []
        }

        // Step 6: Decode codec tokens → waveform
        let outputSamples = numFrames * 1920
        iclLog("  ICL: decoding \(numFrames) frames → \(outputSamples) samples...")
        let waveform = codecDecoder.decode(codes: allCodebooks)
        let t3 = CFAbsoluteTimeGetCurrent()

        // Step 7: Optionally trim the reference echo from the start of the waveform.
        // The model is conditioned on [ref_text + target_text] and produces codec
        // for both, so the raw decoded waveform begins with a regeneration of the
        // reference. Trim by the reference's known codec-frame count: the model
        // re-speaks the reference at ~its own encoded length (refCodes.dim(2)
        // frames), which we know exactly — far more reliable than the old token-
        // ratio estimate, whose generous floor clamped the trim and leaked seconds
        // of reference echo.
        let trimmedWaveform: [Float]
        if trimReference {
            let targetTokenCount = tokenizer.encode(text).count
            let refFrames = refCodes.dim(2)
            trimmedWaveform = Qwen3TTSModel.trimICLReferenceByFrames(
                waveform,
                referenceFrames: refFrames,
                targetTokenCount: targetTokenCount)
            let removed = waveform.count - trimmedWaveform.count
            if removed > 0 {
                iclLog("  ICL: trimmed \(removed) reference samples (~\(String(format: "%.2f", Double(removed)/24000.0))s, refFrames=\(refFrames) tgtTok=\(targetTokenCount)) from output start")
            }
        } else {
            trimmedWaveform = waveform
        }

        let audioDur = Double(trimmedWaveform.count) / 24000.0
        let encTime = String(format: "%.3f", t1-t0)
        let genTime = String(format: "%.3f", t2-t1)
        let msPerStep = String(format: "%.0f", (t2-t1)/Double(max(numFrames, 1))*1000)
        let decTime = String(format: "%.3f", t3-t2)
        let totTime = String(format: "%.3f", t3-t0)
        let audDur = String(format: "%.2f", audioDur)
        let rtf = String(format: "%.2f", (t3-t0)/max(audioDur, 0.001))
        iclLog("  ICL timing: encode=\(encTime)s | generate=\(genTime)s (\(numFrames) steps, \(msPerStep)ms/step) | decode=\(decTime)s | total=\(totTime)s | audio=\(audDur)s | RTF=\(rtf)")

        return trimmedWaveform
    }

    // MARK: - Reference echo trim

    /// Trim the reference reproduction by its known codec-frame count.
    ///
    /// ICL output is `[reference reproduction + target]`. The model re-speaks the
    /// reference transcript at ~its original rate, so the reproduction occupies
    /// about `referenceFrames` codec frames (1920 samples each at 24 kHz) — the
    /// same length as the encoded reference, which we know exactly. That makes the
    /// frame count a far better trim point than the old token-ratio estimate,
    /// whose over-generous `6 frames/token` target floor clamped the trim and left
    /// seconds of reference echo (the seed-ladder leak that failed the grader's
    /// prefix check).
    ///
    /// The reproduction length still varies run-to-run, so the trim is a best
    /// estimate, not exact: a longer-than-encoded reproduction leaves a short tail
    /// (a leading word or two), a shorter one risks clipping the target's first
    /// word. Both are caught by the caller's ASR grader + seed ladder.
    static func trimICLReferenceByFrames(
        _ waveform: [Float],
        referenceFrames: Int,
        targetTokenCount: Int
    ) -> [Float] {
        guard referenceFrames > 0 else { return waveform }
        let samplesPerFrame = 1920
        // ~ the reproduction length + 1 frame for the ref→target boundary word.
        let estimate = (referenceFrames + 1) * samplesPerFrame
        // Safety floor against pathological over-trim: if the reproduction came out
        // much shorter than its encoded length, `estimate` would eat into the
        // target, so cap the trim to keep at least this many target samples. The
        // cap only binds when the reproduction is well under `referenceFrames`; in
        // the normal regime the exact frame `estimate` governs. Keep it loose
        // (~2 frames/token) so it stays a last-resort net and doesn't displace the
        // known-frame estimate with a coarser token-length guess (a higher floor
        // would bind in the common case and re-introduce reference-echo leak).
        let minTarget = max(samplesPerFrame * 4, targetTokenCount * 2 * samplesPerFrame)
        // When the whole take is shorter than the floor (a degenerate/repetitive
        // generation), `trim` collapses to 0 and the untrimmed waveform — reference
        // echo included — is returned for the grader to reject and the seed ladder
        // to retry.
        let trim = min(estimate, max(0, waveform.count - minTarget))
        guard trim > 0, trim < waveform.count else { return waveform }
        return Array(waveform.dropFirst(trim))
    }

    // MARK: - ICL Prefill Construction

    /// Build ICL prefill embeddings following the Python mlx-audio reference.
    ///
    /// Layout:
    /// ```
    /// [role_embed]                                    ← <|im_start|>assistant\n
    /// [tts_pad...tts_bos + codec_prefix]              ← codec prefix overlay
    /// [ref_text + target_text + tts_eos + codec_pad]  ← text overlay
    /// [codec_bos + ref_codec_embeds]                  ← codec ICL context
    /// ```
    private func buildICLPrefillEmbeddings(
        refCodes: MLXArray,         // [1, 16, T_ref]
        referenceText: String,
        targetText: String,
        language: String,
        languageId: Int?,
        speakerEmbed: MLXArray,     // [1, 1024]
        tokenizer: Qwen3Tokenizer
    ) -> (prefillEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray) {
        let hiddenSize = config.talker.hiddenSize

        // 1. Tokenize ref_text and target_text. The tokenizer reports
        // `add_prefix_space: false`, so encoding raw text is equivalent to
        // encoding the chat-template-wrapped string and slicing the template
        // tokens off — verified by comparing token IDs against HF's Python
        // tokenizer on the same inputs.
        let refTextTokens = tokenizer.encode(referenceText)
        let targetTextTokens = tokenizer.encode(targetText)

        // 2. TTS special tokens
        let ttsPadEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsPad)]).expandedDimensions(axis: 0))
        let ttsBosEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsBos)]).expandedDimensions(axis: 0))
        let ttsEosEmbed = talker.embedText(
            MLXArray([Int32(CodecTokens.ttsEos)]).expandedDimensions(axis: 0))

        // 3. Build text_embed: text_projection(ref_text + target_text) + eos
        let combinedTextIds = (refTextTokens + targetTextTokens).map { Int32($0) }
        let combinedTextArray = MLXArray(combinedTextIds).expandedDimensions(axis: 0)
        let textEmbed = talker.embedText(combinedTextArray)  // [1, T_text, D]
        let textEmbedWithEos = concatenated([textEmbed, ttsEosEmbed], axis: 1)
        let textLens = textEmbedWithEos.dim(1)

        // 4. Build codec_embed: codec_bos + sum_of_all_codebook_embeddings(ref_codes)
        let firstCbCodes = refCodes[0..., 0, 0...]  // [1, T_ref]
        var refCodecEmbed = talker.embedCodec(firstCbCodes)  // [1, T_ref, D]
        for i in 0..<(config.codePredictor.numCodeGroups - 1) {
            let cbCodes = refCodes[0..., i + 1, 0...]  // [1, T_ref]
            refCodecEmbed = refCodecEmbed + codePredictor.codecEmbeddings[i](cbCodes)
        }

        let codecBosEmbed = talker.embedCodec(
            MLXArray([Int32(CodecTokens.codecBos)]).expandedDimensions(axis: 0))
        let codecEmbedICL = concatenated([codecBosEmbed, refCodecEmbed], axis: 1)  // [1, T_ref+1, D]
        let codecLens = codecEmbedICL.dim(1)

        // 5. Streaming overlay — matches the qwen-tts reference default
        // (generate_icl_prompt, non_streaming_mode=False). Align the text with the
        // reference codec ELEMENT-WISE (sum) instead of concatenating them, and feed
        // any text beyond the codec length as trailing.
        //
        // The model still re-speaks the reference before the target (the trim below
        // is still required), but with this overlay the reproduction is ~ the
        // reference's own frame count and far more consistent run-to-run, instead of
        // the longer, highly variable reproduction the non-streaming layout (text
        // block, then codec block) produced. That variability is what defeated the
        // token-ratio trim and leaked seconds of reference echo into the output
        // (failing the grader's prefix check → seed-ladder retries). With the
        // overlay + frame-based trim, takes come out clean often enough that the
        // seed ladder reliably lands a clean one, and ~30-40% fewer frames are
        // generated per take.
        let iclInputEmbed: MLXArray
        let iclTrailing: MLXArray
        if textLens > codecLens {
            iclInputEmbed = textEmbedWithEos[0..., 0..<codecLens, 0...] + codecEmbedICL
            iclTrailing = textEmbedWithEos[0..., codecLens..., 0...]
        } else {
            let padLen = codecLens - textLens
            let paddedText = padLen > 0
                ? concatenated([textEmbedWithEos, broadcast(ttsPadEmbed, to: [1, padLen, hiddenSize])], axis: 1)
                : textEmbedWithEos
            iclInputEmbed = paddedText + codecEmbedICL
            iclTrailing = ttsPadEmbed
        }

        // 6. Codec prefix. Two layouts (reference parity):
        //   With language id: [codec_think, codec_think_bos, lang_id, codec_think_eos, pad, bos]
        //                     speaker injected after think_eos → 7 tokens.
        //   Without (auto):   [codec_nothink, codec_think_bos, codec_think_eos, pad, bos]
        //                     speaker injected after think_eos → 6 tokens.
        let codecPrefixTokens: [Int32]
        let speakerInjectAt: Int
        if let langId = languageId {
            codecPrefixTokens = [
                Int32(CodecTokens.codecThink),
                Int32(CodecTokens.codecThinkBos),
                Int32(langId),
                Int32(CodecTokens.codecThinkEos),
                Int32(CodecTokens.codecPad),
                Int32(CodecTokens.codecBos),
            ]
            speakerInjectAt = 4
        } else {
            codecPrefixTokens = [
                Int32(CodecTokens.codecNothink),
                Int32(CodecTokens.codecThinkBos),
                Int32(CodecTokens.codecThinkEos),
                Int32(CodecTokens.codecPad),
                Int32(CodecTokens.codecBos),
            ]
            speakerInjectAt = 3
        }
        let codecPrefixArray = MLXArray(codecPrefixTokens).expandedDimensions(axis: 0)
        var codecPrefixEmbed = talker.embedCodec(codecPrefixArray)

        // Inject speaker embedding right after think_eos.
        let spkEmbedReshaped = speakerEmbed.reshaped([1, 1, hiddenSize])
        let part0 = codecPrefixEmbed[0..., 0..<speakerInjectAt, 0...]
        let part1 = codecPrefixEmbed[0..., speakerInjectAt..., 0...]
        codecPrefixEmbed = concatenated([part0, spkEmbedReshaped, part1], axis: 1)

        let codecSuffixEmbed = talker.embedCodec(
            MLXArray([Int32(CodecTokens.codecPad), Int32(CodecTokens.codecBos)]).expandedDimensions(axis: 0))

        // 7. Role embedding: <|im_start|>assistant\n (3 tokens)
        let imStartId: Int32 = 151644
        let assistantId: Int32 = 77091
        let newlineId: Int32 = 198
        let roleTokens = MLXArray([imStartId, assistantId, newlineId]).expandedDimensions(axis: 0)
        let roleEmbed = talker.embedText(roleTokens)  // [1, 3, D]

        // 8. Build pad/bos prefix (text side overlaid with codec prefix)
        let prefixLen = codecPrefixEmbed.dim(1)
        let padCount = prefixLen - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, hiddenSize])
        let combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)  // [1, prefixLen-1, D]
        let combinedPrefixOverlay = combinedPrefix + codecPrefixEmbed[0..., 0..<(prefixLen - 1), 0...]

        // 9. Full input_embeds: role + codec_prefix + icl_embed
        let inputEmbeds = concatenated(
            [roleEmbed, combinedPrefixOverlay, iclInputEmbed], axis: 1)

        // Trailing text: leftover target text (streaming) or tts_pad. The generation
        // loop feeds this one token per step so the model produces the target codec.
        return (inputEmbeds, iclTrailing, ttsPadEmbed)
    }
}
