import Foundation

/// The bit width and group size a chat checkpoint's quantized layers must be built with,
/// and the one rule for reading them out of a `config.json`.
///
/// Three generations of exports are in circulation and each states quantization differently:
///   - `quantization_bits` / `quantization_group_size` — current, explicit and unambiguous
///   - `"quantization": {"bits": 8, "group_size": 32}` — mlx-community / `mlx_lm` object form
///   - `"quantization": "int8"` — label only; names a width but carries no group size
///
/// and the oldest checkpoints state nothing at all.
///
/// Every chat config in this module resolves through here rather than reading one form
/// directly, because a config that understands only one form does not fail when it meets
/// another — it silently falls back to INT4/64, and mlx-swift's `Module.update(parameters:)`
/// verifies nothing, so the checkpoint's wider rows are installed into narrow layers and the
/// model answers with garbage instead of refusing to load. One copy of the precedence rule
/// is also the only way the three configs stay in agreement as a fourth form appears.
public struct ChatQuantization: Sendable, Equatable {
    public let bits: Int
    public let groupSize: Int

    /// What a checkpoint that states nothing was: this package only ever shipped INT4
    /// before the quantization fields existed.
    public static let defaultLabel = "int4"
    public static let defaultBits = 4
    public static let defaultGroupSize = 64

    public init(bits: Int, groupSize: Int) {
        self.bits = bits
        self.groupSize = groupSize
    }

    /// Explicit numeric fields first, then the nested object, then the width named by the
    /// label, then INT4 / group size 64. A label that names no width MLX can pack (`"bf16"`,
    /// `"none"`) is ignored rather than trusted, and no label carries a group size — the
    /// load-time weight-shape check is what ultimately decides whether the answer was right.
    public static func resolve(
        explicitBits: Int?,
        explicitGroupSize: Int?,
        nestedBits: Int? = nil,
        nestedGroupSize: Int? = nil,
        label: String? = nil
    ) -> ChatQuantization {
        ChatQuantization(
            bits: explicitBits ?? nestedBits ?? label.flatMap(bits(fromLabel:)) ?? defaultBits,
            groupSize: explicitGroupSize ?? nestedGroupSize ?? defaultGroupSize)
    }

    /// Resolve from decoded `config.json` objects, searched in the given order per field.
    /// More than one object because Gemma 4's multimodal export states quantization at the
    /// root while its text fields live under `text_config`, and either may carry it.
    public static func resolve(searching objects: [[String: Any]]) -> ChatQuantization {
        func first<T>(_ read: ([String: Any]) -> T?) -> T? {
            for object in objects {
                if let value = read(object) { return value }
            }
            return nil
        }
        func number(_ key: String, in object: [String: Any]?) -> Int? {
            (object?[key] as? NSNumber)?.intValue
        }
        let nested = first { $0["quantization"] as? [String: Any] }
        return resolve(
            explicitBits: first { number("quantization_bits", in: $0) },
            explicitGroupSize: first { number("quantization_group_size", in: $0) },
            nestedBits: number("bits", in: nested),
            nestedGroupSize: number("group_size", in: nested),
            label: first { $0["quantization"] as? String })
    }

    /// Pull a bit width out of a label such as `"int4"`, `"INT8"`, or `"4bit"`.
    /// Returns nil unless the digits name a width MLX can actually pack.
    public static func bits(fromLabel label: String) -> Int? {
        let digits = label.drop { !$0.isNumber }.prefix { $0.isNumber }
        guard let bits = Int(digits), [2, 3, 4, 5, 6, 8].contains(bits) else { return nil }
        return bits
    }
}
