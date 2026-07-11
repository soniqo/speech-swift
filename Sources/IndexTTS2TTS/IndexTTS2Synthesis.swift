import AudioCommon
import Foundation
import MLX


enum IndexTTS2StageTimer {
    static let enabled = ProcessInfo.processInfo.environment["INDEXTTS2_TIMING"] == "1"

    static func report(_ stage: String, since start: inout CFAbsoluteTime) {
        let now = CFAbsoluteTimeGetCurrent()
        if enabled {
            FileHandle.standardError.write(Data("[itts2] \(stage): \(String(format: "%.2f", now - start))s\n".utf8))
        }
        start = now
    }
}

public struct IndexTTS2SynthesisOptions: Equatable, Sendable {
    public var speakingRate: Float
    public var maxInternalPauseDuration: Float?
    public var s2MelSteps: Int

    public init(
        speakingRate: Float = 1.0,
        maxInternalPauseDuration: Float? = nil,
        s2MelSteps: Int = 15
    ) throws {
        guard speakingRate.isFinite, speakingRate >= 0.5, speakingRate <= 1.5 else {
            throw AudioModelError.invalidConfiguration(
                model: "IndexTTS2",
                reason: "speakingRate must be finite and in [0.5, 1.5].")
        }
        if let maxInternalPauseDuration {
            guard maxInternalPauseDuration.isFinite,
                  maxInternalPauseDuration >= 0.05,
                  maxInternalPauseDuration <= 2.0 else {
                throw AudioModelError.invalidConfiguration(
                    model: "IndexTTS2",
                    reason: "maxInternalPauseDuration must be finite and in [0.05, 2.0].")
            }
        }
        guard s2MelSteps >= 4, s2MelSteps <= 100 else {
            throw AudioModelError.invalidConfiguration(
                model: "IndexTTS2",
                reason: "s2MelSteps must be in [4, 100].")
        }
        self.speakingRate = speakingRate
        self.maxInternalPauseDuration = maxInternalPauseDuration
        self.s2MelSteps = s2MelSteps
    }

    public static let `default` = try! IndexTTS2SynthesisOptions()
}

enum IndexTTS2PauseCompressor {
    static func compress(
        _ samples: [Float],
        sampleRate: Int,
        maxPauseDuration: Float
    ) -> [Float] {
        guard sampleRate > 0, samples.count > sampleRate / 2 else {
            return samples
        }
        let maxPauseSamples = max(1, Int((Float(sampleRate) * maxPauseDuration).rounded()))
        let window = max(1, Int((Float(sampleRate) * 0.02).rounded()))
        let hop = max(1, Int((Float(sampleRate) * 0.01).rounded()))
        guard samples.count > window else {
            return samples
        }

        var rms = [Float]()
        rms.reserveCapacity(samples.count / hop)
        var offset = 0
        while offset + window <= samples.count {
            var sumSquares: Float = 0
            for sample in samples[offset..<offset + window] {
                sumSquares += sample * sample
            }
            rms.append(sqrt(sumSquares / Float(window)))
            offset += hop
        }
        guard !rms.isEmpty else {
            return samples
        }

        let active = rms.filter { $0.isFinite && $0 > 1e-5 }.sorted()
        guard !active.isEmpty else {
            return samples
        }
        let p90 = active[Int(Float(active.count - 1) * 0.9)]
        let threshold = max(Float(0.0015), p90 * 0.08)
        let leadingGuardSamples = Int(Float(sampleRate) * 0.15)
        let trailingGuardStart = samples.count - leadingGuardSamples

        var removalRanges: [Range<Int>] = []
        var silenceStartFrame: Int?
        for frame in 0...rms.count {
            let isSilent = frame < rms.count && rms[frame] < threshold
            if isSilent, silenceStartFrame == nil {
                silenceStartFrame = frame
            } else if !isSilent, let startFrame = silenceStartFrame {
                let startSample = min(samples.count, startFrame * hop)
                let endSample = min(samples.count, frame * hop + window)
                let duration = endSample - startSample
                if startSample > leadingGuardSamples,
                   endSample < trailingGuardStart,
                   duration > maxPauseSamples + hop {
                    let keepBefore = maxPauseSamples / 2
                    let keepAfter = maxPauseSamples - keepBefore
                    let removeStart = min(endSample, startSample + keepBefore)
                    let removeEnd = max(removeStart, endSample - keepAfter)
                    if removeEnd > removeStart {
                        removalRanges.append(removeStart..<removeEnd)
                    }
                }
                silenceStartFrame = nil
            }
        }

        guard !removalRanges.isEmpty else {
            return samples
        }

        var output = [Float]()
        output.reserveCapacity(samples.count - removalRanges.reduce(0) { $0 + $1.count })
        var cursor = samples.startIndex
        for range in removalRanges {
            if cursor < range.lowerBound {
                output.append(contentsOf: samples[cursor..<range.lowerBound])
            }
            cursor = range.upperBound
        }
        if cursor < samples.endIndex {
            output.append(contentsOf: samples[cursor..<samples.endIndex])
        }
        return output
    }
}

extension IndexTTS2NativeRuntime {
    func synthesize(
        textTokens: [Int],
        conditioning: IndexTTS2ReferenceConditioning,
        semanticOptions: IndexTTS2SemanticGenerationOptions = IndexTTS2SemanticGenerationOptions(),
        synthesisOptions: IndexTTS2SynthesisOptions = .default
    ) throws -> [Float] {
        var stageStart = CFAbsoluteTimeGetCurrent()
        let semantic = try semanticGPT.generateSemanticCodes(
            textTokens: textTokens,
            conditioning: conditioning,
            options: semanticOptions)
        IndexTTS2StageTimer.report("semantic-gpt (\(semantic.codeCount) codes)", since: &stageStart)
        return try synthesize(
            textTokens: textTokens,
            conditioning: conditioning,
            semantic: semantic,
            synthesisOptions: synthesisOptions)
    }

    func synthesize(
        textTokens: [Int],
        semanticCodes: [Int32],
        conditioning: IndexTTS2ReferenceConditioning,
        synthesisOptions: IndexTTS2SynthesisOptions = .default
    ) throws -> [Float] {
        let semantic = semanticGPT.semanticGeneration(
            codes: semanticCodes,
            conditioning: conditioning)
        return try synthesize(
            textTokens: textTokens,
            conditioning: conditioning,
            semantic: semantic,
            synthesisOptions: synthesisOptions)
    }

    func synthesize(
        textTokens: [Int],
        conditioning: IndexTTS2ReferenceConditioning,
        semantic: IndexTTS2SemanticGeneration,
        synthesisOptions: IndexTTS2SynthesisOptions = .default
    ) throws -> [Float] {
        guard semantic.codeCount > 0 else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 synthesis",
                reason: "Semantic GPT produced no speech tokens.")
        }

        var latent = semanticGPT.latentForS2Mel(
            textTokens: textTokens,
            generatedCodes: semantic.codeTensor,
            conditioningLatent: semantic.conditioningLatent,
            emotionHidden: conditioning.emotionSemanticHidden,
            emotionVectorOverride: conditioning.emotionVectorOverride,
            emotionVectorOverrideWeightSum: conditioning.emotionVectorOverrideWeightSum)
        latent = s2MelFlow.gptLatent(latent)

        var semanticPrompt = semanticQuantizer.vq2Emb(codes: semantic.codeTensor)
            .transposed(0, 2, 1)
        semanticPrompt = semanticPrompt + latent

        let frameExpansion = 1.72 / synthesisOptions.speakingRate
        let targetLength = max(1, Int((Float(semantic.codeCount) * frameExpansion).rounded(.down)))
        let generatedCondition = lengthRegulator(semanticPrompt, targetLength: targetLength)
        let condition = concatenated([conditioning.promptCondition, generatedCondition], axis: 1)

        var stageStart = CFAbsoluteTimeGetCurrent()
        eval(condition)
        IndexTTS2StageTimer.report("s2mel-prep", since: &stageStart)
        let mel = s2MelFlow.inference(
            condition: condition,
            promptMel: conditioning.promptMel,
            style: conditioning.styleEmbedding,
            steps: synthesisOptions.s2MelSteps)
        eval(mel)
        IndexTTS2StageTimer.report("s2mel-flow", since: &stageStart)
        let promptFrames = conditioning.promptMel.dim(2)
        let generatedMel = mel[0..., 0..., promptFrames..<mel.dim(2)]
        let waveform = vocoder(generatedMel.transposed(0, 2, 1)).asType(.float32)
        eval(waveform)
        IndexTTS2StageTimer.report("bigvgan", since: &stageStart)
        var samples = waveform.asArray(Float.self)
        if let maxPauseDuration = synthesisOptions.maxInternalPauseDuration {
            samples = IndexTTS2PauseCompressor.compress(
                samples,
                sampleRate: config.outputSampleRate,
                maxPauseDuration: maxPauseDuration)
        }
        return samples
    }
}
