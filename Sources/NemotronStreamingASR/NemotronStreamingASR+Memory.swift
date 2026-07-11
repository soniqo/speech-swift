import AudioCommon

extension NemotronStreamingASRModel: ModelMemoryManageable {
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
        // ~612 MB on-disk INT8 bundle; ANE/GPU residency is similar — ~1.2 GB
        // peak RSS during streaming on M5 Pro (encoder mmap + working set).
        return 612 * 1024 * 1024
    }
}
