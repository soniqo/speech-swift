import Foundation

/// One timestamped NVIDIA Audio2Face-3D output frame.
///
/// The v2.3 Mark model emits 301 coefficients per audio window:
/// skin blendshapes, tongue blendshapes, jaw controls, and eye controls.
public struct Audio2Face3DFrame: Codable, Equatable, Sendable {
    public let timeSeconds: Double
    public let coefficients: [Float]
    public let layout: Audio2Face3DCoefficientLayout

    public init(
        timeSeconds: Double,
        coefficients: [Float],
        layout: Audio2Face3DCoefficientLayout = .nvidiaV23Mark
    ) {
        self.timeSeconds = timeSeconds
        self.coefficients = coefficients
        self.layout = layout
    }
}

public struct Audio2Face3DCoefficientLayout: Codable, Equatable, Sendable {
    public let skinCount: Int
    public let tongueCount: Int
    public let jawCount: Int
    public let eyeCount: Int

    public static let nvidiaV23Mark = Audio2Face3DCoefficientLayout(
        skinCount: 272,
        tongueCount: 10,
        jawCount: 15,
        eyeCount: 4)

    public init(skinCount: Int, tongueCount: Int, jawCount: Int, eyeCount: Int) {
        self.skinCount = skinCount
        self.tongueCount = tongueCount
        self.jawCount = jawCount
        self.eyeCount = eyeCount
    }

    public var coefficientCount: Int {
        skinCount + tongueCount + jawCount + eyeCount
    }

    public var skinRange: Range<Int> { 0 ..< skinCount }
    public var tongueRange: Range<Int> { skinRange.upperBound ..< skinRange.upperBound + tongueCount }
    public var jawRange: Range<Int> { tongueRange.upperBound ..< tongueRange.upperBound + jawCount }
    public var eyeRange: Range<Int> { jawRange.upperBound ..< jawRange.upperBound + eyeCount }
}
