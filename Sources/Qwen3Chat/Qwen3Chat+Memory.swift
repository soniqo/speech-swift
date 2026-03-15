import AudioCommon

extension Qwen3ChatModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        generator = nil
        conversationHistory.removeAll()
        systemPromptCached = false
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // INT4 Qwen3-0.6B CoreML ≈ 318 MB on disk
        return 318 * 1024 * 1024
    }
}
