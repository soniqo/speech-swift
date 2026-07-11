import AudioCommon
import MLX

extension Qwen3ASRModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        audioEncoder.clearParameters()
        textDecoder?.clearParameters()
        // Restore the MLX cache limit if we lowered it at load time. This
        // un-leaks the cap from PersonaPlex / multi-model processes that
        // co-load this ASR with a Mimi codec, LLM, or TTS that wants the
        // full default cache budget.
        if let prior = savedMLXCacheLimit {
            MLX.Memory.cacheLimit = prior
            savedMLXCacheLimit = nil
        }
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return audioEncoder.parameterMemoryBytes()
            + (textDecoder?.parameterMemoryBytes() ?? 0)
    }
}
