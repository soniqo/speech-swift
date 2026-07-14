import AudioCommon

extension SpeechEnhancer: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        guard _isLoaded else { return }
        network = nil
        _isLoaded = false
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // DeepFilterNet3 Core ML: 8-bit palettized weights, FP16 compute.
        return 2_200_000
    }
}
