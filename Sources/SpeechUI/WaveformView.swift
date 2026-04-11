#if canImport(SwiftUI)
import SwiftUI

/// Renders `[Float]` PCM samples as a centered waveform.
///
/// Downsamples the input to `barCount` bars by taking the peak magnitude in
/// each bucket. Suitable for both file previews (static buffer) and live audio
/// monitoring (push a sliding window of the most recent samples).
///
/// ```swift
/// WaveformView(samples: pcmFloat32Buffer)
///     .frame(height: 64)
///     .foregroundStyle(.tint)
/// ```
public struct WaveformView: View {
    public let samples: [Float]
    public let barCount: Int
    public let barSpacing: CGFloat
    public let minBarHeightFraction: CGFloat
    public let playhead: Double?

    /// - Parameters:
    ///   - samples: Float32 PCM samples in `[-1, 1]`.
    ///   - barCount: Number of bars to render. The view downsamples to this width.
    ///   - barSpacing: Pixel spacing between bars.
    ///   - minBarHeightFraction: Minimum bar height as a fraction of the view
    ///     height, used so silence still draws a faint line. Default `0.02`.
    ///   - playhead: Optional playback position in `[0, 1]`. When set, bars
    ///     before the playhead use the foreground tint and bars after are dimmed.
    public init(
        samples: [Float],
        barCount: Int = 64,
        barSpacing: CGFloat = 2,
        minBarHeightFraction: CGFloat = 0.02,
        playhead: Double? = nil
    ) {
        self.samples = samples
        self.barCount = max(1, barCount)
        self.barSpacing = barSpacing
        self.minBarHeightFraction = minBarHeightFraction
        self.playhead = playhead
    }

    public var body: some View {
        Canvas { context, size in
            let bars = WaveformView.downsamplePeaks(samples, into: barCount)
            guard !bars.isEmpty else { return }

            let totalSpacing = barSpacing * CGFloat(bars.count - 1)
            let barWidth = max(1, (size.width - totalSpacing) / CGFloat(bars.count))
            let midY = size.height / 2
            let playheadFraction = playhead.map { max(0, min(1, $0)) }

            for (index, peak) in bars.enumerated() {
                let normalized = max(min(peak, 1), 0)
                let barHeight = max(size.height * minBarHeightFraction, CGFloat(normalized) * size.height)
                let x = CGFloat(index) * (barWidth + barSpacing)
                let rect = CGRect(
                    x: x,
                    y: midY - barHeight / 2,
                    width: barWidth,
                    height: barHeight
                )

                let isPast = playheadFraction.map { Double(index) / Double(max(bars.count - 1, 1)) <= $0 } ?? true
                let opacity: Double = isPast ? 1.0 : 0.3
                context.fill(Path(rect), with: .color(.primary.opacity(opacity)))
            }
        }
        .accessibilityLabel("Audio waveform")
    }

    /// Downsample to `count` peak bars by taking the max absolute value in each bucket.
    /// Public so callers can pre-compute static waveforms once and reuse them.
    public static func downsamplePeaks(_ samples: [Float], into count: Int) -> [Float] {
        guard count > 0, !samples.isEmpty else { return [] }
        if samples.count <= count {
            return samples.map { abs($0) }
        }
        let bucketSize = Double(samples.count) / Double(count)
        var peaks: [Float] = []
        peaks.reserveCapacity(count)
        for i in 0..<count {
            let start = Int(Double(i) * bucketSize)
            let end = min(samples.count, Int(Double(i + 1) * bucketSize))
            guard start < end else {
                peaks.append(0)
                continue
            }
            var peak: Float = 0
            for j in start..<end {
                let value = abs(samples[j])
                if value > peak { peak = value }
            }
            peaks.append(peak)
        }
        return peaks
    }
}
#endif
