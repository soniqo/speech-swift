import AudioCommon
import MLX

extension Qwen3ASRModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        audioEncoder.clearParameters()
        textDecoder?.clearParameters()
        mlxCacheLimitLease?.release()
        mlxCacheLimitLease = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return audioEncoder.parameterMemoryBytes()
            + (textDecoder?.parameterMemoryBytes() ?? 0)
    }
}
