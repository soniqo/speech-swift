import Darwin

public struct ProcessMemoryUsage: Sendable {
    public let residentBytes: UInt64
    public let physicalFootprintBytes: UInt64
    public let peakPhysicalFootprintBytes: UInt64
}

/// Returns the current process's resident set and physical footprint, in bytes.
///
/// `residentBytes` preserves the benchmark's historical RSS metric. On macOS,
/// however, it can substantially undercount file-backed MLX allocations such as
/// memory-mapped safetensors. `physicalFootprintBytes` uses `TASK_VM_INFO` and is
/// the better estimate of the process's actual pressure on unified memory.
public func currentProcessMemoryUsage() -> ProcessMemoryUsage {
    let footprint = physicalFootprintUsage()
    return ProcessMemoryUsage(
        residentBytes: currentRSSBytes(),
        physicalFootprintBytes: footprint.current,
        peakPhysicalFootprintBytes: footprint.peak)
}

/// Returns the current process's resident set size, in bytes.
///
/// Uses Mach's `mach_task_basic_info`. This is retained as a historical metric,
/// but it can undercount file-backed MLX allocations. For deployment sizing,
/// use ``currentPhysicalFootprintBytes()``. Returns 0 on error.
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

/// Returns the current process's physical footprint, in bytes.
///
/// This matches the memory-pressure accounting used by macOS and includes
/// file-backed GPU mappings that can be absent from `resident_size`. Returns 0
/// on error.
public func currentPhysicalFootprintBytes() -> UInt64 {
    physicalFootprintUsage().current
}

/// Returns the kernel-recorded physical-footprint high-water mark, in bytes.
///
/// Unlike periodic sampling, this captures transient inference peaks even when
/// MLX clears reusable buffers before returning to the benchmark.
public func peakPhysicalFootprintBytes() -> UInt64 {
    physicalFootprintUsage().peak
}

private func physicalFootprintUsage() -> (current: UInt64, peak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kerr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { p in
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), p, &count)
        }
    }
    guard kerr == KERN_SUCCESS else { return (0, 0) }
    let current = UInt64(info.phys_footprint)
    let recordedPeak = UInt64(max(0, info.ledger_phys_footprint_peak))
    return (current, max(current, recordedPeak))
}
