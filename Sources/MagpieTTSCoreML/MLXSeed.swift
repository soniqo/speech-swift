import MLXRandom

/// Lightweight wrapper around ``MLXRandom.seed`` so files that don't
/// import MLXRandom directly can still seed it from `MagpieTTSCoreML.synthesize`.
enum MagpieMLXSeed {
    static func seed(_ value: UInt64) {
        MLXRandom.seed(value)
    }
}
