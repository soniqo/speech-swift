import AudioCommon
import Foundation
import MLX
import MLXCommon
import SpeechRestoration

public struct IndexTTS2ReferenceConditioning {
    public let speakerInputFeatures: MLXArray
    public let speakerAttentionMask: MLXArray
    public let speakerSemanticHidden: MLXArray
    public let speakerSemanticCodes: MLXArray
    public let speakerSemanticPrompt: MLXArray
    public let promptCondition: MLXArray
    public let emotionInputFeatures: MLXArray
    public let emotionAttentionMask: MLXArray
    public let emotionSemanticHidden: MLXArray
    public let emotionVectorOverride: MLXArray?
    public let emotionVectorOverrideWeightSum: Float
    public let promptMel: MLXArray
    public let styleEmbedding: MLXArray
    public let reference16kSampleCount: Int
    public let reference22kSampleCount: Int
    public let emotion16kSampleCount: Int

    public var speakerInputFeatureShape: [Int] { speakerInputFeatures.shape }
    public var speakerAttentionMaskShape: [Int] { speakerAttentionMask.shape }
    public var speakerSemanticHiddenShape: [Int] { speakerSemanticHidden.shape }
    public var speakerSemanticCodeShape: [Int] { speakerSemanticCodes.shape }
    public var speakerSemanticPromptShape: [Int] { speakerSemanticPrompt.shape }
    public var promptConditionShape: [Int] { promptCondition.shape }
    public var emotionInputFeatureShape: [Int] { emotionInputFeatures.shape }
    public var emotionAttentionMaskShape: [Int] { emotionAttentionMask.shape }
    public var emotionSemanticHiddenShape: [Int] { emotionSemanticHidden.shape }
    public var emotionVectorOverrideShape: [Int]? { emotionVectorOverride?.shape }
    public var promptMelShape: [Int] { promptMel.shape }
    public var styleEmbeddingShape: [Int] { styleEmbedding.shape }
}

public enum IndexTTS2ReferenceConditioningError: Error, LocalizedError, Equatable {
    case referenceTooShort(path: String, sampleRate: Int, minimumSamples: Int)

    public var errorDescription: String? {
        switch self {
        case .referenceTooShort(let path, let sampleRate, let minimumSamples):
            return "IndexTTS2 reference audio is too short for \(sampleRate) Hz feature extraction: \(path) needs at least \(minimumSamples) samples."
        }
    }
}

extension IndexTTS2NativeRuntime {
    func prepareReferenceConditioning(
        referenceAudio: URL,
        emotionReferenceAudio: URL? = nil,
        emotionControl: IndexTTS2EmotionControl? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> IndexTTS2ReferenceConditioning {
        progressHandler?(0.05, "Loading IndexTTS2 reference audio")
        let reference16k = try Self.loadClippedAudio(referenceAudio, sampleRate: 16_000)
        let reference22k = try Self.loadClippedAudio(referenceAudio, sampleRate: config.s2Mel.sampleRate)

        progressHandler?(0.25, "Preparing IndexTTS2 semantic front-end features")
        let speakerSemantic = try Self.semanticFrontEndFeatures(
            reference16k,
            source: referenceAudio,
            sampleRate: 16_000)
        let speakerHidden = normalizeSemanticHidden(
            wav2Vec2BertModel.hiddenState17(inputFeatures: speakerSemantic.features))
        let speakerEncoded = semanticEncoder(speakerHidden)
        let speakerQuantized = semanticQuantizer.quantize(speakerEncoded)

        let emotionURL = emotionControl == nil ? (emotionReferenceAudio ?? referenceAudio) : referenceAudio
        let emotion16k: [Float]
        if emotionURL == referenceAudio {
            emotion16k = reference16k
        } else {
            emotion16k = try Self.loadClippedAudio(emotionURL, sampleRate: 16_000)
        }
        let emotionSemantic = try Self.semanticFrontEndFeatures(
            emotion16k,
            source: emotionURL,
            sampleRate: 16_000)
        let emotionHidden = emotionURL == referenceAudio
            ? speakerHidden
            : normalizeSemanticHidden(
                wav2Vec2BertModel.hiddenState17(inputFeatures: emotionSemantic.features))

        progressHandler?(0.55, "Preparing IndexTTS2 prompt mel")
        let promptMel = Self.promptMel(reference22k, config: config)
        let promptCondition = lengthRegulator(
            speakerQuantized.embeddings,
            targetLength: promptMel.dim(2))

        progressHandler?(0.75, "Preparing IndexTTS2 style embedding")
        let style = campPlusEncoder.inference(reference16k).asType(.float32)
        eval(style)

        let emotionOverride: MLXArray?
        let emotionWeightSum: Float
        if let emotionControl {
            let resolved = explicitEmotionVector(control: emotionControl, style: style)
            emotionOverride = resolved.vector
            emotionWeightSum = resolved.weightSum
        } else {
            emotionOverride = nil
            emotionWeightSum = 0
        }

        progressHandler?(1.0, "IndexTTS2 reference conditioning ready")
        return IndexTTS2ReferenceConditioning(
            speakerInputFeatures: speakerSemantic.features,
            speakerAttentionMask: speakerSemantic.attentionMask,
            speakerSemanticHidden: speakerHidden,
            speakerSemanticCodes: speakerQuantized.codes,
            speakerSemanticPrompt: speakerQuantized.embeddings,
            promptCondition: promptCondition,
            emotionInputFeatures: emotionSemantic.features,
            emotionAttentionMask: emotionSemantic.attentionMask,
            emotionSemanticHidden: emotionHidden,
            emotionVectorOverride: emotionOverride,
            emotionVectorOverrideWeightSum: emotionWeightSum,
            promptMel: promptMel,
            styleEmbedding: style,
            reference16kSampleCount: reference16k.count,
            reference22kSampleCount: reference22k.count,
            emotion16kSampleCount: emotion16k.count)
    }

    private func explicitEmotionVector(
        control: IndexTTS2EmotionControl,
        style: MLXArray
    ) -> (vector: MLXArray, weightSum: Float) {
        let styleVector = style.asType(.float32).asArray(Float.self)
        let speakerRows = speakerMatrix.asType(.float32).asArray(Float.self)
        let emotionRows = emotionMatrix.asType(.float32).asArray(Float.self)
        let weights = control.scaledVector
        let weightSum = control.scaledVectorSum
        let styleDim = 192
        let emotionDim = config.gpt.modelDim
        var output = [Float](repeating: 0, count: emotionDim)
        var offset = 0

        for (bucket, count) in config.emotionBucketCounts.enumerated() {
            defer { offset += count }
            let weight = weights[bucket]
            guard weight > 0 else { continue }

            let row = Self.mostSimilarStyleRow(
                style: styleVector,
                speakerRows: speakerRows,
                start: offset,
                count: count,
                dim: styleDim)
            let sourceOffset = (offset + row) * emotionDim
            for i in 0..<emotionDim {
                output[i] += weight * emotionRows[sourceOffset + i]
            }
        }

        let array = MLXArray(output, [1, emotionDim]).asType(.float32)
        eval(array)
        return (array, weightSum)
    }

    private static func mostSimilarStyleRow(
        style: [Float],
        speakerRows: [Float],
        start: Int,
        count: Int,
        dim: Int
    ) -> Int {
        let styleNorm = max(sqrt(style.reduce(Float(0)) { $0 + $1 * $1 }), Float.ulpOfOne)
        var bestIndex = 0
        var bestScore = -Float.greatestFiniteMagnitude
        for row in 0..<count {
            let rowStart = (start + row) * dim
            var dot: Float = 0
            var rowNormSquared: Float = 0
            for i in 0..<dim {
                let value = speakerRows[rowStart + i]
                dot += style[i] * value
                rowNormSquared += value * value
            }
            let score = dot / (styleNorm * max(sqrt(rowNormSquared), Float.ulpOfOne))
            if score > bestScore {
                bestScore = score
                bestIndex = row
            }
        }
        return bestIndex
    }

    private func normalizeSemanticHidden(_ hidden: MLXArray) -> MLXArray {
        let mean = wavStats["mean"]!.asType(hidden.dtype).reshaped([1, 1, hidden.dim(2)])
        let std = sqrt(wavStats["var"]!.asType(hidden.dtype)).reshaped([1, 1, hidden.dim(2)])
        let normalized = (hidden - mean) / std
        eval(normalized)
        return normalized
    }

    private static func loadClippedAudio(_ url: URL, sampleRate: Int) throws -> [Float] {
        let maxSamples = sampleRate * 15
        let loaded = try AudioFileLoader.load(url: url, targetSampleRate: sampleRate)
        if loaded.count > maxSamples {
            return Array(loaded.prefix(maxSamples))
        }
        return loaded
    }

    private static func semanticFrontEndFeatures(
        _ audio16k: [Float],
        source: URL,
        sampleRate: Int
    ) throws -> (features: MLXArray, attentionMask: MLXArray) {
        guard audio16k.count >= SeamlessM4TFrontEnd.frameLength else {
            throw IndexTTS2ReferenceConditioningError.referenceTooShort(
                path: source.path,
                sampleRate: sampleRate,
                minimumSamples: SeamlessM4TFrontEnd.frameLength)
        }

        let (features, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: audio16k)
        guard frames > 0 else {
            throw IndexTTS2ReferenceConditioningError.referenceTooShort(
                path: source.path,
                sampleRate: sampleRate,
                minimumSamples: SeamlessM4TFrontEnd.frameLength)
        }
        let inputFeatures = MLXArray(features, [1, frames, SeamlessM4TFrontEnd.featureDim])
        let mask = MLXArray([Int32](repeating: 1, count: frames), [1, frames])
        eval(inputFeatures, mask)
        return (inputFeatures, mask)
    }

    private static func promptMel(
        _ audio22k: [Float],
        config: IndexTTS2RuntimeConfig
    ) -> MLXArray {
        let spect = config.s2Mel
        let pad = (spect.nFFT - spect.hopLength) / 2
        let padded = reflectPad(audio22k, pad: pad)
        let mel = SlaneyMel.melSpec(
            samples: padded,
            config: SlaneyMelConfig(
                sampleRate: spect.sampleRate,
                nFft: spect.nFFT,
                hop: spect.hopLength,
                win: spect.winLength,
                nMels: spect.nMels,
                fmin: 0,
                fmax: Float(spect.sampleRate) / 2,
                power: 1.0,
                logMel: true,
                logFloor: 1e-5,
                centerPad: false,
                periodicHann: true))
        let prompt = mel.transposed(1, 0).expandedDimensions(axis: 0).asType(.float32)
        eval(prompt)
        return prompt
    }

    private static func reflectPad(_ samples: [Float], pad: Int) -> [Float] {
        guard pad > 0, samples.count > 1 else { return samples }
        var out = [Float](repeating: 0, count: pad + samples.count + pad)
        for i in 0..<samples.count {
            out[pad + i] = samples[i]
        }

        for i in 0..<pad {
            let src = min(max(pad - i, 0), samples.count - 1)
            out[i] = samples[src]
        }

        let last = samples.count - 1
        for i in 0..<pad {
            let src = max(last - 1 - i, 0)
            out[pad + samples.count + i] = samples[src]
        }
        return out
    }
}
