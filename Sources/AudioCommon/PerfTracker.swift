import Foundation
import Darwin

/// Wall-time + resident-memory tracker for benchmark instrumentation.
///
/// Sample at construction, sample again at checkpoints (peak is tracked),
/// then call `finish()` for a printable snapshot. Output format is grep-able
/// from CI logs so we can diff numbers across runs.
public final class PerfTracker {
    private let label: String
    private let startTime: CFAbsoluteTime
    private let startRss: UInt64
    private var peakRss: UInt64

    public init(_ label: String) {
        self.label = label
        self.startTime = CFAbsoluteTimeGetCurrent()
        self.startRss = currentRSS()
        self.peakRss = self.startRss
    }

    /// Sample current RSS and update peak. Useful between long-running steps.
    @discardableResult
    public func checkpoint(_ name: String? = nil) -> UInt64 {
        let cur = currentRSS()
        if cur > peakRss { peakRss = cur }
        if let name {
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            print(String(format: "[PERF] %@ checkpoint=%@ t=%.0fms rss=%.0fMB",
                         label, name, elapsed, Double(cur) / 1_048_576))
        }
        return cur
    }

    /// Final snapshot. Logs in a format easy to grep from CI / diff.
    public func finish() {
        checkpoint()
        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        let curRss = currentRSS()
        print(String(format:
            "[PERF] %@ total=%.0fms rssStart=%.0fMB rssEnd=%.0fMB rssPeak=%.0fMB rssDelta=%.0fMB",
            label, elapsed,
            Double(startRss) / 1_048_576,
            Double(curRss) / 1_048_576,
            Double(peakRss) / 1_048_576,
            Double(Int64(peakRss) - Int64(startRss)) / 1_048_576))
    }
}

/// Current process resident set size in bytes.
public func currentRSS() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? info.resident_size : 0
}
