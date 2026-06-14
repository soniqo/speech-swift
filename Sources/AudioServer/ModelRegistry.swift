import Foundation

// MARK: - Model registry

/// A single selectable model variant.
///
/// Each row carries the exact identity the server will load:
///   - `name` — canonical name echoed back on session.updated
///   - `engine` — dispatch slot ("qwen3-asr", "parakeet", "kokoro", "hibiki", …)
///   - `modelId` — full HuggingFace model identifier passed to `fromPretrained`
///   - `aliases` — short-form names accepted on session.update (`"qwen3"` → 0.6B INT4 by default)
///   - `kind` — which session slot this variant updates (ASR / TTS / S2S)
///
/// Adding a new variant is one row in `MODEL_REGISTRY`. No new resolver code,
/// no new dispatch arm, no surprises.
public struct ModelVariant: Sendable, Equatable {
    public let name: String
    public let engine: String
    public let modelId: String
    public let aliases: [String]
    public let kind: Kind

    public enum Kind: String, Sendable, Equatable {
        case asr
        case tts
        case s2s
    }
}

/// Every model name the Realtime API accepts.
///
/// Order matters only for aliases that collide — within each kind, the first
/// row that owns a given alias wins. That lets "kokoro" → the default Kokoro
/// variant, "qwen3" → the 0.6B INT4 default, etc.
public let MODEL_REGISTRY: [ModelVariant] = [
    // ─── ASR ───────────────────────────────────────────────────────────────
    .init(name: "qwen3-asr-0.6b-mlx-int4",
          engine: "qwen3-asr",
          modelId: "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
          aliases: ["qwen3", "qwen3-asr", "qwen3-0.6b"],
          kind: .asr),
    .init(name: "qwen3-asr-1.7b-mlx-int8",
          engine: "qwen3-asr",
          modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-8bit",
          aliases: ["qwen3-1.7b", "qwen3-asr-1.7b"],
          kind: .asr),
    .init(name: "qwen3-asr-coreml",
          engine: "qwen3-asr",
          modelId: "aufklarer/Qwen3-ASR-CoreML",
          aliases: ["qwen3-coreml", "qwen3-asr-0.6b-coreml"],
          kind: .asr),
    .init(name: "parakeet-tdt-v3-coreml-int8-30s",
          engine: "parakeet",
          modelId: "aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s",
          aliases: ["parakeet", "parakeet-tdt", "parakeet-tdt-v3"],
          kind: .asr),
    .init(name: "parakeet-tdt-v3-coreml-int8-ios-5s",
          engine: "parakeet",
          modelId: "aufklarer/Parakeet-TDT-v3-CoreML-INT8-iOS-5s",
          aliases: ["parakeet-ios", "parakeet-5s"],
          kind: .asr),
    .init(name: "nemotron-3.5-asr-streaming-0.6b-coreml-int8",
          engine: "nemotron",
          modelId: "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8",
          aliases: ["nemotron", "nemotron-3.5", "nemotron-streaming"],
          kind: .asr),
    .init(name: "omnilingual-asr-ctc-300m-coreml-int8-10s",
          engine: "omnilingual",
          modelId: "aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s",
          aliases: ["omnilingual", "omnilingual-300m", "omnilingual-coreml"],
          kind: .asr),

    // ─── TTS ───────────────────────────────────────────────────────────────
    .init(name: "kokoro-82m-coreml",
          engine: "kokoro",
          modelId: "aufklarer/Kokoro-82M-CoreML",
          aliases: ["kokoro", "kokoro-82m"],
          kind: .tts),
    .init(name: "cosyvoice-3-0.5b-mlx-int4",
          engine: "cosyvoice",
          modelId: "aufklarer/CosyVoice3-0.5B-MLX-4bit",
          aliases: ["cosyvoice", "cosyvoice-3", "cosyvoice-0.5b"],
          kind: .tts),
    .init(name: "voxcpm2-mlx-bf16",
          engine: "voxcpm2",
          modelId: "aufklarer/VoxCPM2-MLX-bf16",
          aliases: ["voxcpm2", "voxcpm2-bf16"],
          kind: .tts),
    .init(name: "voxcpm2-mlx-int8",
          engine: "voxcpm2",
          modelId: "aufklarer/VoxCPM2-MLX-int8",
          aliases: ["voxcpm2-int8"],
          kind: .tts),
    .init(name: "qwen3-tts-0.6b-mlx-int4",
          engine: "qwen3-tts",
          modelId: "aufklarer/Qwen3-TTS-12Hz-0.6B-Base-MLX-4bit",
          // "qwen3" is shared with the ASR variant — bare "qwen3" lands in
          // both slots so naming the family pairs ASR + TTS in one update.
          aliases: ["qwen3", "qwen3-tts", "qwen3-speech", "qwen3-tts-0.6b"],
          kind: .tts),
    // Magpie ships as a fixed bundle today (the `MagpieTTSVariant.int4`
    // default). Listing it here keeps it selectable via the protocol; the
    // dispatch site ignores the modelId because MagpieTTS.fromPretrained
    // builds the model from its own variant enum, not an HF slug.
    .init(name: "magpie-tts-multilingual-mlx-int4",
          engine: "magpie",
          modelId: "aufklarer/Magpie-TTS-Multilingual-MLX-4bit",
          aliases: ["magpie", "magpie-tts"],
          kind: .tts),

    // ─── Speech-to-speech (input audio → output audio in one shot) ─────────
    .init(name: "personaplex-7b-mlx-4bit",
          engine: "personaplex",
          modelId: "aufklarer/PersonaPlex-7B-MLX-4bit",
          aliases: ["personaplex", "personaplex-7b"],
          kind: .s2s),
    .init(name: "personaplex-7b-mlx-8bit",
          engine: "personaplex",
          modelId: "aufklarer/PersonaPlex-7B-MLX-8bit",
          aliases: ["personaplex-8bit"],
          kind: .s2s),
    .init(name: "hibiki-zero-3b-mlx-4bit",
          engine: "hibiki",
          modelId: "aufklarer/Hibiki-Zero-3B-MLX-4bit",
          aliases: ["hibiki", "hibiki-zero", "hibiki-3b"],
          kind: .s2s),
    .init(name: "hibiki-zero-3b-mlx-8bit",
          engine: "hibiki",
          modelId: "aufklarer/Hibiki-Zero-3B-MLX-8bit",
          aliases: ["hibiki-8bit"],
          kind: .s2s),
]

/// Look up a model name (canonical or alias) in the registry.
///
/// Case-insensitive. Returns `nil` for empty / unknown names so the
/// session.update handler can apply forward-compat fall-through (keep the
/// current engines, echo the unknown name back).
public func resolveModelVariant(_ name: String) -> ModelVariant? {
    let lower = name.lowercased()
    guard !lower.isEmpty else { return nil }
    // First pass: exact canonical name match.
    if let exact = MODEL_REGISTRY.first(where: { $0.name == lower }) {
        return exact
    }
    // Second pass: alias match. Within a kind, the first row that owns the
    // alias wins — that's how "qwen3" maps to the 0.6B INT4 default rather
    // than the 1.7B INT8 variant.
    return MODEL_REGISTRY.first(where: { variant in
        variant.aliases.contains(lower)
    })
}

/// Find the default variant for an engine string. Used by the session
/// defaults — we want to point at the lightweight Parakeet/Kokoro variants
/// without hard-coding their HF slugs in `RealtimeSession`.
public func defaultVariant(forEngine engine: String, kind: ModelVariant.Kind) -> ModelVariant {
    if let v = MODEL_REGISTRY.first(where: { $0.engine == engine && $0.kind == kind }) {
        return v
    }
    fatalError("No registered variant for engine '\(engine)' (kind: \(kind.rawValue))")
}
