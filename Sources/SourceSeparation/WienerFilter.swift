import Foundation
import Accelerate

/// Multichannel Wiener soft-mask post-filtering for source separation.
///
/// Refines initial magnitude-mask estimates using per-channel ratio masks.
/// All sources must be estimated simultaneously.
///
/// Reference: Nugraha et al. (2016), "Multichannel audio source separation
/// with deep neural networks"
struct WienerFilter {

    /// Apply Wiener soft-mask filtering to refine source estimates.
    ///
    /// Uses per-channel model outputs for mask computation (left and right
    /// channels get independent masks from their respective model estimates).
    ///
    /// - Parameters:
    ///   - targetSpecsL: Per-target left-channel magnitude estimates [target][T][bins]
    ///   - targetSpecsR: Per-target right-channel magnitude estimates [target][T][bins]
    ///   - mixReal: Left STFT real [T][bins]
    ///   - mixImag: Left STFT imag [T][bins]
    ///   - mixRealR: Right STFT real [T][bins]
    ///   - mixImagR: Right STFT imag [T][bins]
    ///   - eps: Regularization to avoid division by zero
    /// - Returns: Refined magnitude spectrograms per target, per channel
    static func apply(
        targetSpecsL: [[[Float]]],  // [target][T][bins] left channel magnitudes
        targetSpecsR: [[[Float]]],  // [target][T][bins] right channel magnitudes
        mixReal: [[Float]],
        mixImag: [[Float]],
        mixRealR: [[Float]],
        mixImagR: [[Float]],
        eps: Float = 1e-10
    ) -> (leftMag: [[[Float]]], rightMag: [[[Float]]]) {
        let nTargets = targetSpecsL.count
        let T = targetSpecsL[0].count
        let bins = targetSpecsL[0][0].count

        var leftResults = [[[Float]]](repeating: [[Float]](repeating: [Float](repeating: 0, count: bins), count: T), count: nTargets)
        var rightResults = leftResults

        for t in 0..<T {
            for f in 0..<bins {
                // Left channel: mask from left-channel model outputs
                var totalPowerL: Float = eps
                for j in 0..<nTargets {
                    totalPowerL += targetSpecsL[j][t][f] * targetSpecsL[j][t][f]
                }
                let leftMixMag = sqrt(mixReal[t][f] * mixReal[t][f] + mixImag[t][f] * mixImag[t][f])

                // Right channel: mask from right-channel model outputs
                var totalPowerR: Float = eps
                for j in 0..<nTargets {
                    totalPowerR += targetSpecsR[j][t][f] * targetSpecsR[j][t][f]
                }
                let rightMixMag = sqrt(mixRealR[t][f] * mixRealR[t][f] + mixImagR[t][f] * mixImagR[t][f])

                for j in 0..<nTargets {
                    let maskL = (targetSpecsL[j][t][f] * targetSpecsL[j][t][f]) / totalPowerL
                    leftResults[j][t][f] = maskL * leftMixMag

                    let maskR = (targetSpecsR[j][t][f] * targetSpecsR[j][t][f]) / totalPowerR
                    rightResults[j][t][f] = maskR * rightMixMag
                }
            }
        }

        return (leftResults, rightResults)
    }
}
