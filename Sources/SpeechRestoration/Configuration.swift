import Foundation

/// Sidon speech-restoration model variant.
///
/// Both variants share the architecture and front-end; they differ only in the
/// predictor's weight precision on disk (the vocoder stays FP16/FP32 either
/// way). `int8` palettizes the predictor (k-means) for ~half the disk and lower
/// peak RAM at a small naturalness cost; `fp16` is the higher-fidelity default.
public enum SidonVariant: String, Sendable, CaseIterable {
    case fp16
    case int8

    /// Default HuggingFace repo id for this variant.
    ///
    /// Provisional repo (`aufklarer/Sidon-CoreML`); the variant selects a
    /// subfolder inside that repo. Override the whole id via
    /// `SpeechRestorer.fromPretrained(modelId:)` if the published layout differs.
    public var defaultModelId: String { SidonConfig.defaultModelId }

    /// Subfolder within the repo that holds this variant's two `.mlpackage` /
    /// `.mlmodelc` bundles.
    public var subfolder: String {
        switch self {
        case .fp16: return "fp16"
        case .int8: return "int8"
        }
    }
}

/// Configuration for the Sidon restoration pipeline.
///
/// The numbers here are fixed by the export (see
/// `speech-models/models/sidon/export/NOTES.md`): the CoreML predictor and
/// vocoder were traced at a **fixed** sequence length of `frames` (= 499 ≈ 10 s),
/// so the runtime chunks longer audio into `windowSamples`-sized windows.
public struct SidonConfig: Sendable {
    /// Input sample rate for the front-end / predictor (w2v-BERT is 16 kHz).
    public let inputSampleRate: Int
    /// Output sample rate produced by the DAC vocoder.
    public let outputSampleRate: Int
    /// Fixed predictor/vocoder sequence length in stacked frames.
    public let frames: Int
    /// Predictor hidden size (w2v-BERT 2.0 last_hidden_state width).
    public let hiddenSize: Int
    /// Stacked feature dimension (`80 mels * stride 2`).
    public let featureDim: Int
    /// Input samples per fixed window. `frames` stacked frames span this many
    /// 16 kHz samples (499 → 160000 = exactly 10 s).
    public let windowSamples: Int
    /// Output samples the vocoder emits per window (`audio` graph output length).
    public let outputSamplesPerWindow: Int

    /// Provisional HuggingFace repo id holding the CoreML bundles. Parameterized
    /// because the published id may change.
    public static let defaultModelId = "aufklarer/Sidon-CoreML"

    /// File / directory names inside a variant subfolder.
    public static let predictorPackageName = "Sidon-Predictor.mlpackage"
    public static let vocoderPackageName = "Sidon-Vocoder.mlpackage"
    public static let predictorCompiledName = "Sidon-Predictor.mlmodelc"
    public static let vocoderCompiledName = "Sidon-Vocoder.mlmodelc"

    /// Default configuration matching the shipped export.
    public static let `default` = SidonConfig(
        inputSampleRate: 16_000,
        outputSampleRate: 48_000,
        frames: 499,
        hiddenSize: 1024,
        featureDim: 160,
        // 499 stacked frames = 998 mel frames; the extractor yields 499 stacked
        // frames for exactly 160000 input samples (10 s @ 16 kHz).
        windowSamples: 160_000,
        // Vocoder `audio` output length for a 499-frame input (DAC ×960 minus
        // the conv-stack trim): 479014 samples ≈ 9.98 s @ 48 kHz.
        outputSamplesPerWindow: 479_014
    )
}
