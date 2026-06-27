import Foundation
import MLX

public struct FishAudioSamplingConfig: Sendable, Equatable {
    public var maxNewTokens: Int
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float
    public var minNewTokens: Int

    public init(
        maxNewTokens: Int = 1_024,
        temperature: Float = 1.0,
        topK: Int = 30,
        topP: Float = 0.9,
        repetitionPenalty: Float = 1.0,
        minNewTokens: Int = 1
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.minNewTokens = minNewTokens
    }

    public static let `default` = FishAudioSamplingConfig()
    public static let greedy = FishAudioSamplingConfig(
        maxNewTokens: 1_024,
        temperature: 0,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
        minNewTokens: 0)
}

public struct FishAudioGeneratedCodebooks: Sendable, Equatable {
    public let codes: [[Int]]

    public var frameCount: Int { codes.first?.count ?? 0 }
    public var codebookCount: Int { codes.count }
}

enum FishAudioSampler {
    static let repetitionAwareWindow = 10

    static func sample(
        logits: [Float],
        allowedTokenIds: [Int],
        config: FishAudioSamplingConfig,
        previousTokens: [Int] = []
    ) -> Int {
        guard !allowedTokenIds.isEmpty else { return 0 }

        var candidates = allowedTokenIds.filter { $0 >= 0 && $0 < logits.count }
        guard !candidates.isEmpty else { return 0 }

        var adjusted = [Float]()
        adjusted.reserveCapacity(candidates.count)
        let repeated = Set(previousTokens.suffix(64))
        for token in candidates {
            var value = logits[token]
            if config.repetitionPenalty > 1, repeated.contains(token) {
                value = value > 0 ? value / config.repetitionPenalty : value * config.repetitionPenalty
            }
            adjusted.append(value)
        }

        if config.topK > 0, config.topK < candidates.count {
            let ranked = adjusted.indices.sorted { adjusted[$0] > adjusted[$1] }
                .prefix(config.topK)
            candidates = ranked.map { candidates[$0] }
            adjusted = ranked.map { adjusted[$0] }
        }

        if config.temperature <= 0 {
            let best = adjusted.indices.max { adjusted[$0] < adjusted[$1] } ?? 0
            return candidates[best]
        }

        let temperature = max(config.temperature, 1e-5)
        let maxLogit = adjusted.max() ?? 0
        var probs = adjusted.map { exp(($0 - maxLogit) / temperature) }
        let total = probs.reduce(Float(0), +)
        if total > 0 {
            for index in probs.indices { probs[index] /= total }
        }

        var order = probs.indices.sorted { probs[$0] > probs[$1] }
        if config.topP < 1 {
            var cumulative: Float = 0
            var cutoff = order.count
            for (rank, index) in order.enumerated() {
                cumulative += probs[index]
                if cumulative >= config.topP {
                    cutoff = rank + 1
                    break
                }
            }
            order = Array(order.prefix(max(1, cutoff)))
        }

        let keptSum = order.reduce(Float(0)) { $0 + probs[$1] }
        let target = Float.random(in: 0..<1) * max(keptSum, 1e-12)
        var cumulative: Float = 0
        for index in order {
            cumulative += probs[index]
            if cumulative >= target {
                return candidates[index]
            }
        }
        return candidates[order.last ?? 0]
    }

    static func sampleRepetitionAwareSemantic(
        logits: [Float],
        allowedTokenIds: [Int],
        config: FishAudioSamplingConfig,
        previousTokens: [Int],
        semanticRange: ClosedRange<Int>
    ) -> Int {
        let token = sample(
            logits: logits,
            allowedTokenIds: allowedTokenIds,
            config: config,
            previousTokens: previousTokens)
        guard semanticRange.contains(token),
              previousTokens.suffix(repetitionAwareWindow).contains(token) else {
            return token
        }

        let fallbackConfig = FishAudioSamplingConfig(
            maxNewTokens: config.maxNewTokens,
            temperature: 1.0,
            topK: config.topK,
            topP: 0.9,
            repetitionPenalty: 1.0,
            minNewTokens: config.minNewTokens)
        return sample(
            logits: logits,
            allowedTokenIds: allowedTokenIds,
            config: fallbackConfig)
    }
}

public extension FishAudioDualARModel {
    func generateCodebooks(
        from input: FishAudioModelInput,
        sampling: FishAudioSamplingConfig = .default
    ) throws -> FishAudioGeneratedCodebooks {
        guard input.rowCount == config.audioDecoder.numCodebooks + 1 else {
            throw FishAudioError.invalidCodebookShape(
                "input row count \(input.rowCount) does not match model count \(config.audioDecoder.numCodebooks + 1)")
        }
        guard input.tokenCount > 0 else {
            throw FishAudioError.invalidCodebookShape("input prompt must not be empty")
        }

        var slowState = initialSlowState()
        var currentInput = input.asMLXArray()
        var generatedRows = Array(repeating: [Int](), count: config.audioDecoder.numCodebooks)
        var previousMainTokens: [Int] = []
        let semanticCandidates = Array(config.semanticStartTokenId...config.semanticEndTokenId)

        for _ in 0..<sampling.maxNewTokens {
            let (slowResult, nextSlowState) = forwardSlow(inputIds: currentInput, state: slowState)
            slowState = nextSlowState

            let lastIndex = slowResult.logits.dim(1) - 1
            let mainLogits = slowResult.logits[0, lastIndex].asType(.float32).asArray(Float.self)
            let canStop = generatedRows[0].count >= max(0, sampling.minNewTokens)
            let allowedSemanticTokens = canStop ? semanticCandidates + [config.eosTokenId] : semanticCandidates
            let mainToken = FishAudioSampler.sampleRepetitionAwareSemantic(
                logits: mainLogits,
                allowedTokenIds: allowedSemanticTokens,
                config: sampling,
                previousTokens: previousMainTokens,
                semanticRange: config.semanticStartTokenId...config.semanticEndTokenId)
            previousMainTokens.append(mainToken)

            if mainToken == config.eosTokenId {
                break
            }

            var frame = [Int]()
            frame.reserveCapacity(config.audioDecoder.numCodebooks + 1)
            frame.append(mainToken)

            let firstCode = min(
                max(mainToken - config.semanticStartTokenId, 0),
                FishAudioDefaults.codebookSize - 1)
            frame.append(firstCode)

            var fastState = initialFastState()
            let hidden = slowResult.hiddenStates[0, lastIndex, 0...]
                .reshaped([1, 1, config.audioDecoder.hiddenSize])
            let (_, initialFastState) = forwardFast(inputEmbeddings: hidden, state: fastState)
            fastState = initialFastState

            var previousCode = firstCode
            let fastCandidates = Array(0..<config.audioDecoder.vocabSize)
            for _ in 1..<config.audioDecoder.numCodebooks {
                let codeInput = MLXArray([Int32(previousCode)]).reshaped([1, 1])
                let codeEmbedding = fast.embedCodebooks(codeInput)
                let (fastLogits, nextFastState) = forwardFast(
                    inputEmbeddings: codeEmbedding,
                    state: fastState)
                fastState = nextFastState

                let logits = fastLogits[0, fastLogits.dim(1) - 1].asType(.float32).asArray(Float.self)
                let code = FishAudioSampler.sample(
                    logits: logits,
                    allowedTokenIds: fastCandidates,
                    config: sampling)
                frame.append(code)
                previousCode = code
            }

            for codebook in 0..<config.audioDecoder.numCodebooks {
                generatedRows[codebook].append(frame[codebook + 1])
            }
            currentInput = MLXArray(frame.map(Int32.init))
                .reshaped([1, config.audioDecoder.numCodebooks + 1, 1])
        }

        return FishAudioGeneratedCodebooks(codes: generatedRows)
    }
}
