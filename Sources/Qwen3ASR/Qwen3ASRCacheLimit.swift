import Foundation
import MLX

/// Owns the process-wide MLX cache ceiling for all loaded Qwen3-ASR models.
/// Every model registers, even when another model or the caller already
/// configured a lower limit. Releasing a registration can then preserve
/// the remaining models' ceilings regardless of unload order.
internal final class Qwen3ASRCacheLimitCoordinator: @unchecked Sendable {
    static let shared = Qwen3ASRCacheLimitCoordinator(
        getLimit: { MLX.Memory.cacheLimit },
        setLimit: { MLX.Memory.cacheLimit = $0 })

    private let lock = NSLock()
    private let getLimit: () -> Int
    private let setLimit: (Int) -> Void
    private var ceilings: [UUID: Int] = [:]
    private var baselineLimit: Int?
    private var appliedLimit: Int?

    /// Injectable accessors keep ownership tests independent of Metal.
    init(getLimit: @escaping () -> Int, setLimit: @escaping (Int) -> Void) {
        self.getLimit = getLimit
        self.setLimit = setLimit
    }

    func acquire(ceiling: Int) -> Lease? {
        guard ceiling > 0 else { return nil }
        lock.lock()
        defer { lock.unlock() }

        if ceilings.isEmpty {
            baselineLimit = getLimit()
            appliedLimit = baselineLimit
        }
        let id = UUID()
        ceilings[id] = ceiling
        updateLimit()
        return Lease(coordinator: self, id: id)
    }

    private func release(id: UUID) {
        lock.lock()
        defer { lock.unlock() }
        guard ceilings.removeValue(forKey: id) != nil else { return }
        updateLimit()
        if ceilings.isEmpty {
            baselineLimit = nil
            appliedLimit = nil
        }
    }

    /// A limit changed outside this coordinator becomes the new caller
    /// budget. In particular, unload must not undo a caller's tighter cap.
    private func updateLimit() {
        let current = getLimit()
        if current != appliedLimit {
            baselineLimit = current
        }
        let baseline = baselineLimit ?? current
        let target = min(baseline, ceilings.values.min() ?? baseline)
        if target != current {
            setLimit(target)
        }
        appliedLimit = target
    }

    final class Lease: @unchecked Sendable {
        private let coordinator: Qwen3ASRCacheLimitCoordinator
        private let id: UUID

        fileprivate init(coordinator: Qwen3ASRCacheLimitCoordinator, id: UUID) {
            self.coordinator = coordinator
            self.id = id
        }

        /// Idempotent, so explicit unload and later destruction both work.
        func release() {
            coordinator.release(id: id)
        }

        deinit {
            release()
        }
    }
}
