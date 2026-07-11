import Foundation
import Darwin.Mach

/// Background sampler for peak process memory (`phys_footprint`, matching
/// Xcode's Memory Report / Instruments). Poll on a dedicated thread so a
/// synchronous CoreML inference on the calling thread can't starve it.
final class MemoryMonitor: @unchecked Sendable {
    private let lock = NSLock()
    private var _peak: Double = 0
    private var running = false

    var peakMB: Double {
        lock.lock(); defer { lock.unlock() }
        return _peak
    }

    /// Current resident footprint in MB.
    static func currentMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { raw in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), raw, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / (1024 * 1024)
    }

    func reset() {
        lock.lock(); _peak = 0; lock.unlock()
    }

    func start() {
        lock.lock(); running = true; lock.unlock()
        let thread = Thread { [weak self] in
            while true {
                guard let self else { return }
                self.lock.lock()
                let go = self.running
                self.lock.unlock()
                if !go { return }
                let m = MemoryMonitor.currentMB()
                self.lock.lock()
                if m > self._peak { self._peak = m }
                self.lock.unlock()
                usleep(20_000) // ~50 Hz
            }
        }
        thread.stackSize = 512 * 1024
        thread.start()
    }

    func stop() {
        lock.lock(); running = false; lock.unlock()
    }
}
