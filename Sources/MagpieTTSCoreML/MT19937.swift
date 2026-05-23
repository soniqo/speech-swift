// Adapted from FluidInference/FluidAudio (Apache-2.0)
// https://github.com/FluidInference/FluidAudio/blob/main/Sources/FluidAudio/TTS/Magpie/Shared/MagpieMT19937.swift
//
// NumPy-compatible Mersenne Twister so that seeding the sampler matches
// `np.random.seed(seed)` in the FluidInference reference pipeline (and
// therefore the upstream NeMo reference). Without bit-identical RNG, our
// `--seed 42` output diverges from the reference even when every other
// numeric path is correct, which makes regression testing painful.

import Foundation

public final class MagpieCoreMLMT19937: RandomNumberGenerator {
    private static let n = 624
    private static let m = 397
    private static let upperMask: UInt32 = 0x8000_0000
    private static let lowerMask: UInt32 = 0x7FFF_FFFF
    private static let matrixA: UInt32 = 0x9908_B0DF

    private var mt: [UInt32] = Array(repeating: 0, count: MagpieCoreMLMT19937.n)
    private var mti: Int = MagpieCoreMLMT19937.n

    public init(seed: UInt32) {
        initGenrand(seed)
    }

    private func initGenrand(_ s: UInt32) {
        mt[0] = s
        for i in 1..<Self.n {
            mt[i] = 1_812_433_253 &* (mt[i - 1] ^ (mt[i - 1] >> 30)) &+ UInt32(i)
        }
        mti = Self.n
    }

    public func genrandInt32() -> UInt32 {
        if mti >= Self.n {
            let mag01: [UInt32] = [0, Self.matrixA]
            var kk = 0
            while kk < Self.n - Self.m {
                let y = (mt[kk] & Self.upperMask) | (mt[kk + 1] & Self.lowerMask)
                mt[kk] = mt[kk + Self.m] ^ (y >> 1) ^ mag01[Int(y & 1)]
                kk += 1
            }
            while kk < Self.n - 1 {
                let y = (mt[kk] & Self.upperMask) | (mt[kk + 1] & Self.lowerMask)
                mt[kk] = mt[kk &+ (Self.m - Self.n)] ^ (y >> 1) ^ mag01[Int(y & 1)]
                kk += 1
            }
            let yLast = (mt[Self.n - 1] & Self.upperMask) | (mt[0] & Self.lowerMask)
            mt[Self.n - 1] = mt[Self.m - 1] ^ (yLast >> 1) ^ mag01[Int(yLast & 1)]
            mti = 0
        }
        var y = mt[mti]
        mti += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= (y >> 18)
        return y
    }

    /// 53-bit precision uniform draw in `[0, 1)` (matches NumPy `genrand_res53`).
    public func uniformDouble() -> Double {
        let a = UInt64(genrandInt32() >> 5)
        let b = UInt64(genrandInt32() >> 6)
        return (Double(a) * 67_108_864.0 + Double(b)) * (1.0 / 9_007_199_254_740_992.0)
    }

    public func next() -> UInt64 {
        let lo = UInt64(genrandInt32())
        let hi = UInt64(genrandInt32())
        return (hi << 32) | lo
    }
}

extension MagpieCoreMLMT19937 {
    /// Reproduces `np.random.choice(len(probs), p=probs)` semantics.
    public func numpyChoice(probs: [Double]) -> Int {
        precondition(!probs.isEmpty, "numpyChoice requires non-empty probability vector")
        var cdf = [Double](repeating: 0, count: probs.count)
        var total: Double = 0
        for i in 0..<probs.count {
            let p = probs[i] > 0 ? probs[i] : 0
            total += p
            cdf[i] = total
        }
        if total <= 0 { return probs.count - 1 }
        let u = uniformDouble() * total
        var lo = 0
        var hi = cdf.count
        while lo < hi {
            let mid = (lo &+ hi) >> 1
            if cdf[mid] > u { hi = mid } else { lo = mid + 1 }
        }
        return Swift.min(lo, probs.count - 1)
    }
}
