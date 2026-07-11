import Foundation
import MLX

/// Multichannel Wiener EM post-filtering for stereo source separation —
/// MLX-vectorized port of `WienerFilter`.
///
/// Mirrors the reference algorithm bit-for-bit (initial complex y from
/// magnitude × normalized mix complex, E-step PSD + SCM, M-step 2×2 complex
/// Wiener gain), but expresses every per-(t, f, source) operation as a
/// broadcast MLX op so the GPU runs the inner loops.
///
/// API matches `WienerFilter.apply` so the two can be swapped behind a flag.
struct WienerFilterMLX {

    /// MLX-native result of Wiener filtering. Shapes are `[J, T, F]` so the
    /// downstream iSTFT path can run on the GPU without any CPU round-trip.
    struct ResultMLX {
        let realL: MLXArray   // [J, T, F]
        let imagL: MLXArray
        let realR: MLXArray
        let imagR: MLXArray
    }

    /// MLX-native Wiener — returns the refined complex spectrum per source as
    /// MLXArrays so callers can keep the data on the GPU through the iSTFT.
    static func applyMLX(
        targetMagsL: [[[Float]]],
        targetMagsR: [[[Float]]],
        mixRealL: [[Float]],
        mixImagL: [[Float]],
        mixRealR: [[Float]],
        mixImagR: [[Float]],
        iterations: Int = 1,
        windowLen: Int = 300
    ) -> ResultMLX {
        let nSources = targetMagsL.count
        let T = targetMagsL[0].count
        let nBins = targetMagsL[0][0].count

        let tgtL = MLXArray(flatten3(targetMagsL), [nSources, T, nBins])
        let tgtR = MLXArray(flatten3(targetMagsR), [nSources, T, nBins])
        let mxRL = MLXArray(flatten2(mixRealL), [T, nBins])
        let mxIL = MLXArray(flatten2(mixImagL), [T, nBins])
        let mxRR = MLXArray(flatten2(mixRealR), [T, nBins])
        let mxIR = MLXArray(flatten2(mixImagR), [T, nBins])

        // Per-window EM. Each window writes into a slice of the per-source
        // output tensors; final result is [J, T, F].
        var perWindowRL: [MLXArray] = []
        var perWindowIL: [MLXArray] = []
        var perWindowRR: [MLXArray] = []
        var perWindowIR: [MLXArray] = []

        var pos = 0
        while pos < T {
            let end = min(T, pos + windowLen)
            let wT = end - pos

            let tL = tgtL[0..., pos..<end, 0...]
            let tR = tgtR[0..., pos..<end, 0...]
            let mRL = mxRL[pos..<end, 0...]
            let mIL = mxIL[pos..<end, 0...]
            let mRR = mxRR[pos..<end, 0...]
            let mIR = mxIR[pos..<end, 0...]

            let refined = emWienerWindow(
                tgtL: tL, tgtR: tR,
                mxRL: mRL, mxIL: mIL, mxRR: mRR, mxIR: mIR,
                wT: wT, nBins: nBins, nSources: nSources, iterations: iterations)
            perWindowRL.append(refined.rL)
            perWindowIL.append(refined.iL)
            perWindowRR.append(refined.rR)
            perWindowIR.append(refined.iR)
            pos = end
        }

        // Concatenate windows along time axis. No eval here — leave the
        // graph lazy so the caller can fuse the iSTFT into the same dispatch.
        return ResultMLX(
            realL: concatenated(perWindowRL, axis: 1),
            imagL: concatenated(perWindowIL, axis: 1),
            realR: concatenated(perWindowRR, axis: 1),
            imagR: concatenated(perWindowIR, axis: 1))
    }

    /// Apply windowed multichannel Wiener EM filtering. `[[Float]]` API
    /// retained for the equivalence unit test and legacy callers.
    static func apply(
        targetMagsL: [[[Float]]],
        targetMagsR: [[[Float]]],
        mixRealL: [[Float]],
        mixImagL: [[Float]],
        mixRealR: [[Float]],
        mixImagR: [[Float]],
        iterations: Int = 1,
        windowLen: Int = 300
    ) -> [(realL: [[Float]], imagL: [[Float]], realR: [[Float]], imagR: [[Float]])] {
        let result = applyMLX(
            targetMagsL: targetMagsL, targetMagsR: targetMagsR,
            mixRealL: mixRealL, mixImagL: mixImagL,
            mixRealR: mixRealR, mixImagR: mixImagR,
            iterations: iterations, windowLen: windowLen)

        eval(result.realL, result.imagL, result.realR, result.imagR)
        let nSources = targetMagsL.count
        let T = targetMagsL[0].count
        let nBins = targetMagsL[0][0].count

        let rLFlat = result.realL.asArray(Float.self)
        let iLFlat = result.imagL.asArray(Float.self)
        let rRFlat = result.realR.asArray(Float.self)
        let iRFlat = result.imagR.asArray(Float.self)

        var out: [(realL: [[Float]], imagL: [[Float]], realR: [[Float]], imagR: [[Float]])] = []
        out.reserveCapacity(nSources)
        for j in 0..<nSources {
            var rL = [[Float]](repeating: [Float](repeating: 0, count: nBins), count: T)
            var iL = rL, rR = rL, iR = rL
            for t in 0..<T {
                let base = (j * T + t) * nBins
                for f in 0..<nBins {
                    rL[t][f] = rLFlat[base + f]
                    iL[t][f] = iLFlat[base + f]
                    rR[t][f] = rRFlat[base + f]
                    iR[t][f] = iRFlat[base + f]
                }
            }
            out.append((rL, iL, rR, iR))
        }
        return out
    }

    // MARK: - Single-window EM in MLX

    private struct WindowResult {
        let rL: MLXArray; let iL: MLXArray; let rR: MLXArray; let iR: MLXArray  // [J, wT, F]
    }

    private static func emWienerWindow(
        tgtL: MLXArray, tgtR: MLXArray,
        mxRL: MLXArray, mxIL: MLXArray, mxRR: MLXArray, mxIR: MLXArray,
        wT: Int, nBins: Int, nSources: Int, iterations: Int
    ) -> WindowResult {
        let eps: Float = 1e-10

        // Numerical scaling — keep peak magnitude bounded.
        let mLmag = sqrt(mxRL * mxRL + mxIL * mxIL)        // [wT, F]
        let mRmag = sqrt(mxRR * mxRR + mxIR * mxIR)
        let maxMagBoth = MLX.maximum(mLmag.max(), mRmag.max()).item(Float.self)
        let scaleDiv = max(Float(1.0), maxMagBoth / 10.0)
        let invScale = 1.0 / scaleDiv

        // Unit-complex direction of the mixture (per t, f).
        let cosL = mxRL / MLX.maximum(mLmag, MLXArray(eps))     // [wT, F]
        let sinL = mxIL / MLX.maximum(mLmag, MLXArray(eps))
        let cosR = mxRR / MLX.maximum(mRmag, MLXArray(eps))
        let sinR = mxIR / MLX.maximum(mRmag, MLXArray(eps))

        // Initial complex estimates: y = target_mag * unit_complex_mix * invScale.
        // Shape after broadcast: [J, wT, F]. Each "y" has 4 real components
        // (real_L, imag_L, real_R, imag_R) — we keep them as 4 separate
        // tensors so the SCM math stays element-wise.
        let s = MLXArray(invScale)
        var yLR = tgtL * cosL * s   // y_L real
        var yLI = tgtL * sinL * s   // y_L imag
        var yRR = tgtR * cosR * s   // y_R real
        var yRI = tgtR * sinR * s   // y_R imag

        // Pre-scaled mixture (used in the M-step gain application).
        let xLR = mxRL * s; let xLI = mxIL * s
        let xRR = mxRR * s; let xRI = mxIR * s

        for _ in 0..<iterations {
            // E-step PSD: v[j, t, f] = 0.5 * (|y_L|^2 + |y_R|^2)
            let v = 0.5 * (yLR * yLR + yLI * yLI + yRR * yRR + yRI * yRI)  // [J, wT, F]

            // E-step SCM per (j, f): R = sum_t (y outer y*) / sum_t v
            //   R[0]   = sum_t |y_L|^2          (real)
            //   R[1,2] = sum_t y_L * conj(y_R)  (complex, real & imag)
            //   R[3]   = sum_t |y_R|^2          (real)
            // The conjugate-mirrored off-diagonal is implicit (R[1] = R[2]*).
            let aRe = yLR; let aIm = yLI
            let bRe = yRR; let bIm = yRI
            let r00Sum = (aRe * aRe + aIm * aIm).sum(axis: 1)              // [J, F]
            let rOffRe = (aRe * bRe + aIm * bIm).sum(axis: 1)              // [J, F]
            let rOffIm = (aIm * bRe - aRe * bIm).sum(axis: 1)              // [J, F]
            let r11Sum = (bRe * bRe + bIm * bIm).sum(axis: 1)              // [J, F]
            let sumV = v.sum(axis: 1) + MLXArray(eps)                       // [J, F]

            let R00 = r00Sum / sumV          // [J, F]   real
            let R01re = rOffRe / sumV
            let R01im = rOffIm / sumV
            let R11 = r11Sum / sumV

            // M-step: Cxx = sum_j v * R  (mixture covariance per (t, f))
            // v has shape [J, wT, F]; R has shape [J, F]; broadcast across wT.
            let vR00 = v * R00[0..., .newAxis, 0...]                        // [J, wT, F]
            let vR01re = v * R01re[0..., .newAxis, 0...]
            let vR01im = v * R01im[0..., .newAxis, 0...]
            let vR11 = v * R11[0..., .newAxis, 0...]
            let c00 = vR00.sum(axis: 0) + MLXArray(eps)                     // [wT, F]
            let c01re = vR01re.sum(axis: 0)
            let c01im = vR01im.sum(axis: 0)
            let c11 = vR11.sum(axis: 0) + MLXArray(eps)
            // Off-diagonal is the conjugate of (c01re, c01im) — call it c10.
            let c10re = c01re
            let c10im = -c01im

            // Inverse of the 2×2 complex matrix Cxx.
            // det = c00*c11 - c01*c10  (complex)
            let detRe = (c00 * c11) - (c01re * c10re - c01im * c10im)
            let detIm = -(c01re * c10im + c01im * c10re)
            let detMag2 = detRe * detRe + detIm * detIm + MLXArray(eps * eps)
            let idR = detRe / detMag2
            let idI = -detIm / detMag2

            // inv(Cxx) = (1/det) * [[c11, -c01], [-c10, c00]]
            let i0r = c11 * idR; let i0i = c11 * idI                         // real entry → still real with imag from idI
            let i1r = -(c01re * idR - c01im * idI)
            let i1i = -(c01re * idI + c01im * idR)
            let i2r = -(c10re * idR - c10im * idI)
            let i2i = -(c10re * idI + c10im * idR)
            let i3r = c00 * idR; let i3i = c00 * idI

            // Per source, build the gain matrix G_j = v_j * R_j  (2×2 complex,
            // with v_j broadcast across the Hermitian R_j) and compute
            // W_j = G_j @ inv(Cxx). Then y_j = W_j @ mix.
            let g0r = v * R00[0..., .newAxis, 0...]                         // [J, wT, F]
            let g0iZ: MLXArray = .zeros(like: g0r)
            let g1r = v * R01re[0..., .newAxis, 0...]
            let g1i = v * R01im[0..., .newAxis, 0...]
            // R[1,0] is the conjugate of R[0,1].
            let g2r = v * R01re[0..., .newAxis, 0...]
            let g2i = -v * R01im[0..., .newAxis, 0...]
            let g3r = v * R11[0..., .newAxis, 0...]
            let g3iZ: MLXArray = .zeros(like: g3r)

            // W = G @ inv(Cxx). Each entry uses complex multiply-add.
            let w0r = (g0r * i0r - g0iZ * i0i) + (g1r * i2r - g1i * i2i)
            let w0i = (g0r * i0i + g0iZ * i0r) + (g1r * i2i + g1i * i2r)
            let w1r = (g0r * i1r - g0iZ * i1i) + (g1r * i3r - g1i * i3i)
            let w1i = (g0r * i1i + g0iZ * i1r) + (g1r * i3i + g1i * i3r)
            let w2r = (g2r * i0r - g2i * i0i) + (g3r * i2r - g3iZ * i2i)
            let w2i = (g2r * i0i + g2i * i0r) + (g3r * i2i + g3iZ * i2r)
            let w3r = (g2r * i1r - g2i * i1i) + (g3r * i3r - g3iZ * i3i)
            let w3i = (g2r * i1i + g2i * i1r) + (g3r * i3i + g3iZ * i3r)

            // Apply W to the (pre-scaled) mixture to produce the refined y.
            yLR = w0r * xLR - w0i * xLI + w1r * xRR - w1i * xRI
            yLI = w0r * xLI + w0i * xLR + w1r * xRI + w1i * xRR
            yRR = w2r * xLR - w2i * xLI + w3r * xRR - w3i * xRI
            yRI = w2r * xLI + w2i * xLR + w3r * xRI + w3i * xRR
        }

        // Scale back to the original signal range.
        return WindowResult(
            rL: yLR * scaleDiv,
            iL: yLI * scaleDiv,
            rR: yRR * scaleDiv,
            iR: yRI * scaleDiv)
    }

    // MARK: - Array flattening helpers

    private static func flatten3(_ x: [[[Float]]]) -> [Float] {
        var out = [Float](); out.reserveCapacity(x.count * x[0].count * x[0][0].count)
        for j in 0..<x.count {
            for t in 0..<x[j].count {
                out.append(contentsOf: x[j][t])
            }
        }
        return out
    }

    private static func flatten2(_ x: [[Float]]) -> [Float] {
        var out = [Float](); out.reserveCapacity(x.count * x[0].count)
        for row in x { out.append(contentsOf: row) }
        return out
    }
}
