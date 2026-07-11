import Foundation
import ArgumentParser
import MLX
import MLXFast
import AudioCommon
import StableAudio3MusicGen

/// Compare Swift SA3 inference against a Python reference dump produced by
/// `models/stable-audio-3/export/scripts/dump_parity_ref.py`. Per stage,
/// prints cosine similarity + max-abs-diff vs the reference tensors.
///
/// Stages: T5Gemma encode • conditioner • DiT single-step velocity • SAME-L decode.
public struct SA3ParityCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "sa3-parity",
        abstract: "Compare Swift SA3 outputs to a Python reference dump (per stage)."
    )

    @Option(name: .long, help: "Reference .safetensors dumped by scripts/dump_parity_ref.py")
    public var ref: String

    @Option(name: .long, help: "Local SA3 bundle directory (skip HF download).")
    public var bundle: String

    @Option(name: .long, help: "Prompt to run through Swift T5Gemma (must match Python dump).")
    public var prompt: String = "lofi house loop"

    @Option(name: .long, help: "Seconds to use when computing T_lat (must match Python).")
    public var seconds: Float = 4.0

    public init() {}

    public func run() throws {
        try runAsync {
            try await self.runImpl()
        }
    }

    private func runImpl() async throws {
        let refURL = URL(fileURLWithPath: ref)
        let bundleURL = URL(fileURLWithPath: bundle)

        guard FileManager.default.fileExists(atPath: refURL.path) else {
            throw ValidationError("--ref not found: \(refURL.path)")
        }

        print("Loading Swift bundle from \(bundle)…"); fflush(stdout)
        let tLat = StableAudio3MusicGen.computeTLat(seconds: seconds)
        let model = try await StableAudio3MusicGen.fromPretrained(
            variant: .mediumInt8,
            tLatHint: tLat,
            localBundleOverride: bundleURL,
            progressHandler: nil)
        print("  bundle loaded"); fflush(stdout)

        print("\nLoading reference: \(ref)"); fflush(stdout)
        let refArrs = try MLX.loadArrays(url: refURL)
        print("  ref loaded: \(refArrs.count) tensors"); fflush(stdout)

        // ─── Stage 1: T5Gemma encode ─────────────────────────────────────
        print("\n[1] T5Gemma encode"); fflush(stdout)
        // (a) Swift tokenizer parity
        let swiftIds = model.t5.tokenizer.encodeAsIds(prompt)
        let refIdsArr = refArrs["input_ids"]!
        // first `mask.sum()` ids are the actual non-pad tokens
        let refMask = refArrs["mask"]!
        let refKeep = Int(refMask.sum().item(Int32.self))
        let refIdsAll = refIdsArr.asType(DType.int32).asArray(Int32.self)
        let refIds = Array(refIdsAll.prefix(refKeep)).map { Int($0) }
        print("  swift tokenizer: \(swiftIds.count) ids — \(swiftIds.prefix(8))")
        print("  python tokenizer: \(refIds.count) ids — \(refIds.prefix(8))")
        print("  ids match: \(swiftIds == refIds)")
        fflush(stdout)

        // (b) Encoder forward with PYTHON token ids (isolate tokenizer bug from encoder bugs)
        print("  [1a] encoder forward using python ids…"); fflush(stdout)
        let pyEmbeds = model.t5.encoder(refIdsArr, attentionMask: refMask)
        eval(pyEmbeds)
        compare(label: "embeds_pyIds", swift: pyEmbeds.asType(DType.float32),
                ref: refArrs["embeds"]!.asType(DType.float32))

        // (c) Encoder forward with SWIFT token ids (combined: tokenizer + encoder)
        print("  [1b] encoder forward using swift ids…"); fflush(stdout)
        let (embedsSwift, _) = model.t5.encode(prompt)
        eval(embedsSwift)
        compare(label: "embeds_swIds", swift: embedsSwift.asType(DType.float32),
                ref: refArrs["embeds"]!.asType(DType.float32))

        // ─── Stage 2: Conditioner ────────────────────────────────────────
        print("\n[2] Conditioner (SecondsTotalEmbedder + apply_prompt_padding)")
        let secsSwift = model.secondsEmbedder(seconds).asType(DType.float32)
        eval(secsSwift)
        compare(label: "secs_emb", swift: secsSwift,
                ref: refArrs["seconds_embed"]!.asType(DType.float32))

        // ─── Stage 3: DiT single-step velocity ───────────────────────────
        print("\n[3] DiT single-step velocity")
        let noise = refArrs["noise"]!.asType(DType.float16)
        let crossAttn = refArrs["cross_attn"]!.asType(DType.float16)
        let globalCond = refArrs["global_cond"]!.asType(DType.float16)
        let t = refArrs["t"]!.asType(DType.float16)
        let vSwift = model.dit(noise, t: t,
                                crossAttnCondRaw: crossAttn,
                                globalCondRaw: globalCond, localAddCond: nil)
        eval(vSwift)
        compare(label: "v", swift: vSwift.asType(DType.float32),
                ref: refArrs["v"]!.asType(DType.float32))

        // ─── Stage 4: SAME-L decoder ─────────────────────────────────────
        print("\n[4] SAME-L decode (denoised → patches)")
        let denoised = refArrs["denoised"]!.asType(DType.float32)
        let patchesSwift = model.decoder(denoised)
        eval(patchesSwift)
        compare(label: "patches", swift: patchesSwift.asType(DType.float32),
                ref: refArrs["patches"]!.asType(DType.float32))
    }

    /// Print cosine similarity + max-abs-diff for two same-shape MLX arrays.
    private func compare(label: String, swift: MLXArray, ref: MLXArray) {
        print("    [compare \(label)] enter; swift \(swift.shape) ref \(ref.shape)"); fflush(stdout)
        let a = swift.reshaped([-1]).asType(DType.float32)
        let b = ref.reshaped([-1]).asType(DType.float32)
        eval(a, b)
        let dot = (a * b).sum()
        let normA = sqrt((a * a).sum())
        let normB = sqrt((b * b).sum())
        eval(dot, normA, normB)
        let cosV = (dot / (normA * normB + MLXArray(Float(1e-12)))).item(Float.self)
        let maxV = (a - b).abs().max().item(Float.self)
        let aMean = a.mean().item(Float.self)
        let bMean = b.mean().item(Float.self)
        let aStd  = sqrt(((a - aMean) * (a - aMean)).mean()).item(Float.self)
        let bStd  = sqrt(((b - bMean) * (b - bMean)).mean()).item(Float.self)
        let shape = swift.shape == ref.shape ? "\(swift.shape)" : "\(swift.shape) vs ref \(ref.shape)"
        let verdict = cosV > 0.99 ? "✓" : cosV > 0.90 ? "~" : "✗"
        print("    \(verdict) \(label.padding(toLength: 12, withPad: " ", startingAt: 0)) "
            + String(format: "cos=%.5f  max_abs=%.4f   swift μ=%+.4f σ=%.4f   ref μ=%+.4f σ=%.4f   shape=%@",
                     cosV, maxV, aMean, aStd, bMean, bStd, shape))
        fflush(stdout)
    }
}

/// Helper: run an async block synchronously and return the result.
func runAsyncReturning<T>(_ block: @escaping () async throws -> T) throws -> T {
    let sem = DispatchSemaphore(value: 0)
    var result: Result<T, Error>!
    Task {
        do { result = .success(try await block()) }
        catch { result = .failure(error) }
        sem.signal()
    }
    sem.wait()
    return try result.get()
}
