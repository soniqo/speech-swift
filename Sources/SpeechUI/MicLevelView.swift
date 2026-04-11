#if canImport(SwiftUI)
import SwiftUI

/// Horizontal level meter for live mic monitoring.
///
/// Takes a normalized level in `[0, 1]` (typically the RMS of the most recent
/// mic chunk) and renders a smooth bar with a peak-hold indicator. The peak
/// hold decays at `peakDecayPerSecond` per second.
///
/// ```swift
/// MicLevelView(level: viewModel.micLevel)
///     .frame(width: 120, height: 8)
/// ```
public struct MicLevelView: View {
    public let level: Double
    public let peakDecayPerSecond: Double
    public let cornerRadius: CGFloat

    public init(
        level: Double,
        peakDecayPerSecond: Double = 0.6,
        cornerRadius: CGFloat = 2
    ) {
        self.level = max(0, min(1, level))
        self.peakDecayPerSecond = peakDecayPerSecond
        self.cornerRadius = cornerRadius
    }

    @State private var peakLevel: Double = 0
    @State private var peakUpdatedAt: Date = .distantPast

    public var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(.secondary.opacity(0.2))

                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(.tint)
                    .frame(width: geo.size.width * level)
                    .animation(.easeOut(duration: 0.08), value: level)

                Rectangle()
                    .fill(.primary)
                    .frame(width: 2, height: geo.size.height)
                    .offset(x: max(0, geo.size.width * peakLevel - 1))
                    .opacity(peakLevel > 0 ? 0.8 : 0)
            }
            .onChange(of: level) { _, newValue in
                let now = Date()
                let elapsed = now.timeIntervalSince(peakUpdatedAt)
                let decayed = max(0, peakLevel - peakDecayPerSecond * elapsed)
                if newValue >= decayed {
                    peakLevel = newValue
                    peakUpdatedAt = now
                } else {
                    peakLevel = decayed
                }
            }
        }
        .accessibilityLabel("Microphone level")
        .accessibilityValue("\(Int(level * 100)) percent")
    }
}

public extension MicLevelView {
    /// Compute an RMS level in `[0, 1]` from a chunk of Float32 PCM samples.
    /// Convenience for the common case where you want to plug raw mic samples
    /// directly into the view.
    static func rmsLevel(samples: [Float]) -> Double {
        guard !samples.isEmpty else { return 0 }
        var sumSquares: Double = 0
        for s in samples {
            sumSquares += Double(s) * Double(s)
        }
        let rms = (sumSquares / Double(samples.count)).squareRoot()
        return min(1, rms)
    }
}
#endif
