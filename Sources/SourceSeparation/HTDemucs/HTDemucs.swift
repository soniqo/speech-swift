import Foundation
import MLX
import MLXNN

/// Hybrid Transformer Demucs (Demucs v4) — a single sub-model. Assembles the
/// parallel frequency (Conv2d U-Net) and time (Conv1d U-Net) branches, merged
/// by a cross-domain transformer at the bottleneck, then recombined as
/// iSTFT(freq-CaC) + time. Translated from demucs htdemucs.py + the MIT MLX
/// reference. NOT yet parity-validated (Phase D).
///
/// For htdemucs_ft (depth 4, nfft 4096) the freq axis never reduces to 1, so
/// there are no empty/inject merge layers — both branches run in parallel.
final class HTDemucs: Module {
    let cfg: HTDemucsConfig
    let depth: Int
    let nfft: Int
    let cac: Bool
    let freqEmbScale: Float
    let bottomChannels: Int
    let trainingLength: Int

    let spec: HTDemucsSpec

    @ModuleInfo(key: "encoder") var encoder: [HEncLayer]
    @ModuleInfo(key: "decoder") var decoder: [HDecLayer]
    @ModuleInfo(key: "tencoder") var tencoder: [HEncLayer]
    @ModuleInfo(key: "tdecoder") var tdecoder: [HDecLayer]
    @ModuleInfo(key: "freq_emb") var freqEmb: ScaledEmbedding?
    @ModuleInfo(key: "channel_upsampler") var channelUpsampler: Conv1d?
    @ModuleInfo(key: "channel_downsampler") var channelDownsampler: Conv1d?
    @ModuleInfo(key: "channel_upsampler_t") var channelUpsamplerT: Conv1d?
    @ModuleInfo(key: "channel_downsampler_t") var channelDownsamplerT: Conv1d?
    @ModuleInfo(key: "crosstransformer") var crosstransformer: CrossTransformerEncoder?

    init(_ cfg: HTDemucsConfig) {
        self.cfg = cfg
        let a = cfg.arch
        self.depth = a.depth
        self.nfft = a.nfft
        self.cac = a.cac
        self.freqEmbScale = a.freqEmb
        self.bottomChannels = a.bottomChannels
        self.trainingLength = cfg.trainingLength
        self.spec = HTDemucsSpec(nfft: a.nfft, hop: a.nfft / 4)

        let S = cfg.sources.count
        var enc: [HEncLayer] = [], dec: [HDecLayer] = []
        var tenc: [HEncLayer] = [], tdec: [HDecLayer] = []
        var fEmb: ScaledEmbedding? = nil

        var chin = cfg.audioChannels
        var chinZ = cac ? chin * 2 : chin
        var chout = a.channels
        var choutZ = a.channels
        var freqs = a.nfft / 2

        for index in 0..<depth {
            let freq = freqs > 1
            var ker = freq ? a.kernelSize : a.timeStride * 2
            var stri = freq ? a.stride : a.timeStride
            var padBool = true
            var lastFreq = false
            if freq && freqs <= a.kernelSize { ker = freqs; padBool = false; lastFreq = true }
            let pad = padBool ? ker / 4 : 0
            // time-branch kernel/stride are always kernel_size/stride (kwt)
            let tKer = a.kernelSize, tStri = a.stride, tPad = a.kernelSize / 4

            if lastFreq { choutZ = max(chout, choutZ); chout = choutZ }

            enc.append(HEncLayer(
                chin: chinZ, chout: choutZ, kernelSize: ker, stride: stri, pad: pad,
                freq: freq, empty: false, rewrite: a.rewrite, context: a.contextEnc,
                dconv: (a.dconvMode & 1) != 0 ? DConv(channels: choutZ, depth: a.dconvDepth, compress: a.dconvComp) : nil))
            if freq {
                tenc.append(HEncLayer(
                    chin: chin, chout: chout, kernelSize: tKer, stride: tStri, pad: tPad,
                    freq: false, empty: lastFreq, rewrite: a.rewrite, context: a.contextEnc,
                    dconv: (a.dconvMode & 1) != 0 ? DConv(channels: chout, depth: a.dconvDepth, compress: a.dconvComp) : nil))
            }

            if index == 0 { chin = cfg.audioChannels * S; chinZ = cac ? chin * 2 : chin }

            dec.insert(HDecLayer(
                chin: choutZ, chout: chinZ, kernelSize: ker, stride: stri, pad: pad,
                last: index == 0, freq: freq, empty: false, rewrite: a.rewrite, context: a.context,
                dconv: (a.dconvMode & 2) != 0 ? DConv(channels: choutZ, depth: a.dconvDepth, compress: a.dconvComp) : nil), at: 0)
            if freq {
                tdec.insert(HDecLayer(
                    chin: chout, chout: chin, kernelSize: tKer, stride: tStri, pad: tPad,
                    last: index == 0, freq: false, empty: lastFreq, rewrite: a.rewrite, context: a.context,
                    dconv: (a.dconvMode & 2) != 0 ? DConv(channels: chout, depth: a.dconvDepth, compress: a.dconvComp) : nil), at: 0)
            }

            chin = chout; chinZ = choutZ
            chout = a.growth * chout; choutZ = a.growth * choutZ
            if freq { freqs = freqs <= a.kernelSize ? 1 : freqs / a.stride }
            if index == 0 && a.freqEmb > 0 {
                fEmb = ScaledEmbedding(freqs, chinZ, scale: Float(a.embScale))
            }
        }

        self._encoder.wrappedValue = enc
        self._decoder.wrappedValue = dec
        self._tencoder.wrappedValue = tenc
        self._tdecoder.wrappedValue = tdec
        self._freqEmb.wrappedValue = fEmb

        let transformerChannels = a.channels * Int(pow(Double(a.growth), Double(depth - 1)))
        if a.bottomChannels > 0 {
            self._channelUpsampler.wrappedValue = Conv1d(inputChannels: transformerChannels, outputChannels: a.bottomChannels, kernelSize: 1)
            self._channelDownsampler.wrappedValue = Conv1d(inputChannels: a.bottomChannels, outputChannels: transformerChannels, kernelSize: 1)
            self._channelUpsamplerT.wrappedValue = Conv1d(inputChannels: transformerChannels, outputChannels: a.bottomChannels, kernelSize: 1)
            self._channelDownsamplerT.wrappedValue = Conv1d(inputChannels: a.bottomChannels, outputChannels: transformerChannels, kernelSize: 1)
        } else {
            self._channelUpsampler.wrappedValue = nil
            self._channelDownsampler.wrappedValue = nil
            self._channelUpsamplerT.wrappedValue = nil
            self._channelDownsamplerT.wrappedValue = nil
        }
        let dim = a.bottomChannels > 0 ? a.bottomChannels : transformerChannels
        if a.tLayers > 0 {
            self._crosstransformer.wrappedValue = CrossTransformerEncoder(
                dim: dim, nhead: a.tHeads, ffn: Int(a.tHiddenScale * Float(dim)),
                numLayers: a.tLayers, maxPeriod: a.tMaxPeriod, weightPosEmbed: a.tWeightPosEmbed)
        } else {
            self._crosstransformer.wrappedValue = nil
        }
        super.init()
    }

    private func centerTrim(_ x: MLXArray, length: Int) -> MLXArray {
        let delta = x.dim(-1) - length
        if delta <= 0 { return x }
        let front = delta / 2
        return x[.ellipsis, front ..< (front + length)]
    }

    /// `mix`: [B, audioChannels, T] → stems [B, S, audioChannels, T].
    func callAsFunction(_ mixIn: MLXArray) -> MLXArray {
        var mix = mixIn
        let length = mix.dim(-1)
        var lengthPrePad: Int? = nil
        if mix.dim(-1) < trainingLength {
            lengthPrePad = mix.dim(-1)
            mix = padded(mix, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)),
                                       IntOrPair((0, trainingLength - lengthPrePad!))])
        }

        let z = spec.spec(mix)                       // complex [B, C, Fq, le]
        var x = spec.magnitudeCaC(z)                 // [B, 2C, Fq, le]
        let B = x.dim(0), Fq = x.dim(2), T = x.dim(3)
        let mean = x.mean(axes: [1, 2, 3], keepDims: true)
        let std = sqrt(x.variance(axes: [1, 2, 3], keepDims: true))
        x = (x - mean) / (1e-5 + std)

        var xt = mix
        let meant = xt.mean(axes: [1, 2], keepDims: true)
        let stdt = sqrt(xt.variance(axes: [1, 2], keepDims: true))
        xt = (xt - meant) / (1e-5 + stdt)

        var saved: [MLXArray] = [], savedT: [MLXArray] = []
        var lengths: [Int] = [], lengthsT: [Int] = []
        for idx in 0..<encoder.count {
            lengths.append(x.dim(-1))
            var inject: MLXArray? = nil
            if idx < tencoder.count {
                lengthsT.append(xt.dim(-1))
                let te = tencoder[idx]
                xt = te(xt)
                if te.empty { inject = xt } else { savedT.append(xt) }
            }
            x = encoder[idx](x, inject: inject)
            if idx == 0, let fEmb = freqEmb {
                let Fr = x.dim(2)
                let frs = MLXArray(Array(0..<Fr).map { Int32($0) })
                let emb = fEmb(frs).transposed(1, 0).reshaped([1, x.dim(1), Fr, 1])
                x = x + freqEmbScale * emb
            }
            saved.append(x)
        }

        if let ct = crosstransformer {
            if bottomChannels > 0 {
                let c = x.dim(1), f = x.dim(2), t = x.dim(3)
                x = applyConv1dNCL(x.reshaped([x.dim(0), c, f * t]), channelUpsampler!).reshaped([x.dim(0), bottomChannels, f, t])
                xt = applyConv1dNCL(xt, channelUpsamplerT!)
                let (nx, nxt) = ct(x, xt)
                x = applyConv1dNCL(nx.reshaped([nx.dim(0), bottomChannels, f * t]), channelDownsampler!).reshaped([nx.dim(0), c, f, t])
                xt = applyConv1dNCL(nxt, channelDownsamplerT!)
            } else {
                (x, xt) = ct(x, xt)
            }
        }

        let offset = depth - tdecoder.count
        for idx in 0..<decoder.count {
            let skip = saved.removeLast()
            let (nx, pre) = decoder[idx](x, skip: skip, length: lengths.removeLast())
            x = nx
            if idx >= offset {
                let td = tdecoder[idx - offset]
                let lt = lengthsT.removeLast()
                if td.empty {
                    let (nxt, _) = td(pre[0..., 0..., 0, 0...], skip: nil, length: lt)
                    xt = nxt
                } else {
                    let st = savedT.removeLast()
                    let (nxt, _) = td(xt, skip: st, length: lt)
                    xt = nxt
                }
            }
        }

        let S = cfg.sources.count
        x = x * std + mean                            // denorm (per-batch scalar)
        let zout = spec.maskCaC(x, sources: S, channels: cfg.audioChannels)  // complex [B,S,C,Fq,T]
        var out = spec.ispec(zout.reshaped([B * S, cfg.audioChannels, Fq, T]), length: trainingLength)
        out = out.reshaped([B, S, cfg.audioChannels, trainingLength])

        let actualLen = xt.dim(-1)
        var xtOut = (xt * stdt + meant).reshaped([B, S, cfg.audioChannels, actualLen])
        out = centerTrim(out, length: actualLen)
        out = xtOut + out
        out = out[.ellipsis, 0 ..< min(out.dim(-1), trainingLength)]
        if let lp = lengthPrePad { out = out[.ellipsis, 0 ..< lp] }
        return out
    }
}
