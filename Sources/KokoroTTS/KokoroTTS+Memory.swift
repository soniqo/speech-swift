import AudioCommon

extension KokoroTTSModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        network = nil
        voiceEmbeddings = [:]
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // ~86 MB for INT8 quantized model
        return 86 * 1024 * 1024
    }
}
