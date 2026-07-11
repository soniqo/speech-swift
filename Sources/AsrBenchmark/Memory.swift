import Darwin

/// Returns the current process's resident set size, in bytes.
///
/// Uses Mach's `task_info` — same backing as Activity Monitor's
/// "Memory" column — so the value reflects real physical-page footprint
/// (not virtual address space). Returns 0 on error.
public func currentRSSBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kerr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { p in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), p, &count)
        }
    }
    guard kerr == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size)
}
