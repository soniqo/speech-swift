import MLX
import XCTest

@testable import Qwen3TTS

final class CodePredictorFrameTests: XCTestCase {
    private func tinyConfig(embeddingDim: Int = 8) -> CodePredictorConfig {
        var config = CodePredictorConfig()
        config.hiddenSize = 8
        config.embeddingDim = embeddingDim
        config.numLayers = 2
        config.numHeads = 2
        config.numKVHeads = 1
        config.headDim = 4
        config.intermediateSize = 16
        config.vocabSize = 32
        config.numCodeGroups = 4
        config.bits = 0
        return config
    }

    private func assertCompiledFrameMatchesStepwiseGreedyPrediction(
        config: CodePredictorConfig,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let predictor = CodePredictorModel(config: config)
        let input = MLXArray((0..<(2 * config.embeddingDim)).map { Float($0) / 20 })
            .reshaped([1, 2, config.embeddingDim])

        var (logits, cache) = predictor(
            inputsEmbeds: input,
            groupIndex: 0,
            cache: nil)
        var token = argMax(logits[0..., 1..<2, 0...].squeezed()).asType(.int32)
        var expectedTokens = [token]
        for groupIndex in 1..<(config.numCodeGroups - 1) {
            let embedding = predictor.embedCodecGroup(
                token.reshaped(1, 1),
                groupIndex: groupIndex - 1)
            (logits, cache) = predictor(
                inputsEmbeds: embedding,
                groupIndex: groupIndex,
                cache: cache)
            token = argMax(logits.squeezed()).asType(.int32)
            expectedTokens.append(token)
        }
        let expected = stacked(expectedTokens)

        let compiled = compile(
            inputs: [predictor],
            outputs: [predictor],
            shapeless: false
        ) { inputs in
            [predictor.predictFrame(
                inputsEmbeds: inputs[0],
                temperature: inputs[1],
                gumbels: inputs[2],
                topK: 0,
                greedy: true)]
        }
        let actual = compiled([
            input,
            MLXArray(Float(0)),
            MLXArray.zeros([config.numCodeGroups - 1, config.vocabSize]),
        ])[0]
        eval(actual, expected)

        XCTAssertEqual(actual.shape, [config.numCodeGroups - 1], file: file, line: line)
        XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self), file: file, line: line)
    }

    func testCompiledFrameMatchesStepwiseGreedyPrediction() {
        assertCompiledFrameMatchesStepwiseGreedyPrediction(config: tinyConfig())
    }

    func testCompiledFrameMatchesProjectedStepwiseGreedyPrediction() {
        assertCompiledFrameMatchesStepwiseGreedyPrediction(config: tinyConfig(embeddingDim: 12))
    }

    func testCompiledFrameMatchesStepwiseTopKPrediction() {
        let config = tinyConfig()
        let predictor = CodePredictorModel(config: config)
        let input = MLXArray((0..<(2 * config.embeddingDim)).map { Float($0) / 20 })
            .reshaped([1, 2, config.embeddingDim])
        let temperature = Float(0.7)
        let topK = 5
        let gumbels = MLXArray(
            (0..<((config.numCodeGroups - 1) * config.vocabSize)).map {
                Float(($0 * 17) % 31) / 10 - 1.5
            })
            .reshaped([config.numCodeGroups - 1, config.vocabSize])

        var (logits, cache) = predictor(inputsEmbeds: input, groupIndex: 0, cache: nil)
        var token = sampledToken(
            logits[0..., 1..<2, 0...],
            temperature: temperature,
            gumbel: gumbels[0],
            topK: topK)
        var expectedTokens = [token]
        for groupIndex in 1..<(config.numCodeGroups - 1) {
            let embedding = predictor.embedCodecGroup(
                token.reshaped(1, 1),
                groupIndex: groupIndex - 1)
            (logits, cache) = predictor(
                inputsEmbeds: embedding,
                groupIndex: groupIndex,
                cache: cache)
            token = sampledToken(
                logits,
                temperature: temperature,
                gumbel: gumbels[groupIndex],
                topK: topK)
            expectedTokens.append(token)
        }
        let expected = stacked(expectedTokens)

        let compiled = compile(
            inputs: [predictor],
            outputs: [predictor],
            shapeless: false
        ) { inputs in
            [predictor.predictFrame(
                inputsEmbeds: inputs[0],
                temperature: inputs[1],
                gumbels: inputs[2],
                topK: topK,
                greedy: false)]
        }
        let actual = compiled([input, MLXArray(temperature), gumbels])[0]
        eval(actual, expected)

        XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self))
    }

    private func sampledToken(
        _ logits: MLXArray,
        temperature: Float,
        gumbel: MLXArray,
        topK: Int
    ) -> MLXArray {
        var scores = logits.squeezed().asType(.float32) / temperature
        let sorted = MLX.sorted(scores)
        let threshold = sorted[scores.dim(0) - topK]
        scores = MLX.where(scores .< threshold, MLXArray(Float(-1e9)), scores)
        return argMax(scores + gumbel).asType(.int32)
    }
}
