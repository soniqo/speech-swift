import Foundation

/// Test-only DSP helpers for synthesizing harsher audio fixtures than the
/// clean studio clips that ship in `Tests/.../Resources/`. Reproduces failure
/// modes the bundled fixtures don't:
///   • Overlapped voices  — forces ASR to disambiguate competing speech
///   • Continuous stitch  — long-form decode without sentence boundaries
///   • Babble background  — conference-room background chatter
///   • White-Gaussian SNR — controlled additive noise
///
/// All helpers operate on `[Float]` PCM; caller is responsible for the sample
/// rate (mix only buffers at the same rate).
///
/// MIRRORED across `NemotronStreamingASRTests/`, `Qwen3ASRTests/`,
/// `OmnilingualASRTests/`. Keep them in sync if you change the API.
enum HarshAudio {
    /// Sample-wise sum `a + b * 10^(gainBdB / 20)`. The shorter buffer is
    /// zero-padded to match the longer; use for overlaying two utterances.
    static func overlay(_ a: [Float], _ b: [Float], gainBdB: Float = 0) -> [Float] {
        let gain = powf(10, gainBdB / 20)
        let n = max(a.count, b.count)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let av = i < a.count ? a[i] : 0
            let bv = i < b.count ? b[i] : 0
            out[i] = av + bv * gain
        }
        return out
    }

    /// Concatenate with `paddingSamples` zeros between buffers. Use `0` for
    /// dense continuous speech with no inter-utterance silence — the worst
    /// case for a long-form decoder's sentence-boundary heuristics.
    static func stitch(_ buffers: [[Float]], paddingSamples: Int = 0) -> [Float] {
        var out = [Float]()
        let pad = [Float](repeating: 0, count: max(0, paddingSamples))
        for (i, b) in buffers.enumerated() {
            out.append(contentsOf: b)
            if i < buffers.count - 1 && paddingSamples > 0 {
                out.append(contentsOf: pad)
            }
        }
        return out
    }

    /// Deterministic white-Gaussian noise (Box-Muller on a seeded xorshift
    /// RNG). Output samples are N(0, 1). Seeded so CI reproduces locally.
    static func whiteNoise(samples: Int, seed: UInt64 = 0x1234_abcd) -> [Float] {
        var rng = SeededRNG(seed: seed)
        var out = [Float](repeating: 0, count: samples)
        var i = 0
        while i < samples {
            let u1 = max(Float(rng.next()) / Float(UInt32.max), 1e-9)
            let u2 = Float(rng.next()) / Float(UInt32.max)
            let r = sqrtf(-2 * logf(u1))
            let theta = 2 * .pi * u2
            out[i] = r * cosf(theta)
            if i + 1 < samples { out[i + 1] = r * sinf(theta) }
            i += 2
        }
        return out
    }

    /// Multi-talker babble: sum `voiceCount` cyclically-shifted copies of
    /// `speech`, then RMS-normalize. Simulates conference-room chatter.
    /// (Real babble uses unrelated speakers; this is a serviceable proxy
    /// that needs no extra fixtures.)
    static func babbleFromSpeech(_ speech: [Float], voiceCount: Int = 8) -> [Float] {
        guard !speech.isEmpty else { return speech }
        var out = [Float](repeating: 0, count: speech.count)
        for i in 0..<voiceCount {
            let offset = (i * 1741 + 311) % speech.count
            for j in 0..<speech.count {
                let src = (j + offset) % speech.count
                out[j] += speech[src]
            }
        }
        let r = rms(out)
        if r > 0 {
            for i in 0..<out.count { out[i] /= r }
        }
        return out
    }

    /// Mix `signal` and `noise` at a target SNR (dB). Noise RMS is scaled to
    /// `RMS(signal) / 10^(snrDB / 20)`; shorter noise is tiled cyclically.
    static func mixAtSNR(signal: [Float], noise: [Float], snrDB: Float) -> [Float] {
        let sigRms = rms(signal)
        let noiseRms = rms(noise)
        guard sigRms > 0, noiseRms > 0 else { return signal }
        let targetNoiseRms = sigRms / powf(10, snrDB / 20)
        let scale = targetNoiseRms / noiseRms
        var out = [Float](repeating: 0, count: signal.count)
        for i in 0..<signal.count {
            out[i] = signal[i] + noise[i % noise.count] * scale
        }
        return out
    }

    private static func rms(_ buf: [Float]) -> Float {
        guard !buf.isEmpty else { return 0 }
        var sumSq: Float = 0
        for s in buf { sumSq += s * s }
        return sqrtf(sumSq / Float(buf.count))
    }
}

private struct SeededRNG {
    var state: UInt64
    init(seed: UInt64) { self.state = seed == 0 ? 0xdead_beef : seed }
    mutating func next() -> UInt32 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return UInt32(truncatingIfNeeded: state)
    }
}
