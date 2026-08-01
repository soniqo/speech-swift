/// Runs an autoregressive generation range with a cancellation checkpoint
/// before every token step.
///
/// The body returns `false` when generation has reached its natural terminal
/// condition. A throwing checkpoint stops the loop before the next token step
/// begins, bounding cooperative cancellation latency to at most one in-flight
/// token step.
@inline(__always)
func runQwen3TTSGenerationLoop(
    _ steps: Range<Int>,
    checkCancellation: () throws -> Void,
    body: (Int) throws -> Bool
) rethrows {
    for step in steps {
        try checkCancellation()
        guard try body(step) else { return }
    }
}
