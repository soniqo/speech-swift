import AudioCommon

extension ParakeetStreamingASRModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        encoder = nil
        decoder = nil
        joint = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        return 150 * 1024 * 1024  // ~150 MB for INT8 120M model
    }
}
