import XCTest
import MLX
import MLXNN
@testable import GLiNER
@testable import GLiNERBenchmark

final class GLiNERTests: XCTestCase {
    func testBenchmarkArguments() throws {
        let args = ["--model","model","--cases","cases.json","--output","out.json"]
        XCTAssertEqual(try BenchmarkOptions.parse(args).iterations,5)
        XCTAssertEqual(try BenchmarkOptions.parse(args+["--iterations","10"]).iterations,10)
        XCTAssertThrowsError(try BenchmarkOptions.parse([]))
        XCTAssertThrowsError(try BenchmarkOptions.parse(args+["--iterations","0"]))
        XCTAssertThrowsError(try BenchmarkOptions.parse(args+["--iterations","bad"]))
        XCTAssertThrowsError(try BenchmarkOptions.parse(args+["--unknown","x"]))
        XCTAssertEqual(try BenchmarkOptions.parse(args+["--cache-limit-mb","64"]).cacheLimitMB,64)
        XCTAssertEqual(try BenchmarkOptions.parse(args+["--evaluate-layers","false"]).evaluateLayers,false)
        XCTAssertThrowsError(try BenchmarkOptions.parse(args+["--cache-limit-mb","-1"]))
        XCTAssertThrowsError(try BenchmarkOptions.parse(args+["--evaluate-layers","yes"]))
    }
    func testSchemaRejectsEmptyAndRepeatedLabels() {
        XCTAssertThrowsError(try GLiNER.validateLabels([]))
        XCTAssertThrowsError(try GLiNER.validateLabels(["a","a"]))
        XCTAssertThrowsError(try GLiNER.validateLabels([" "]))
        XCTAssertNoThrow(try GLiNER.validateLabels(["yes","no"]))
    }
    func testWordOffsetsPreserveUnicodeAndPunctuation() {
        let text = "Call İvan 👋 at 6 PM."
        let words = GLiNER.words(text)
        XCTAssertEqual(words.map { (text as NSString).substring(with: $0.range) },["Call","İvan","👋","at","6","PM","."])
    }
    func testConfigurationAndMissingWeights() throws {
        let json = #"{"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":2,"intermediate_size":16,"vocab_size":32,"position_buckets":8,"max_position_embeddings":32,"layer_norm_eps":0.0000001,"relative_attention":true,"share_att_key":true,"position_biased_input":false,"pos_att_type":["p2c","c2p"],"norm_rel_ebd":"layer_norm","type_vocab_size":0}"#
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(GLiNERConfig.self,from: Data(json.utf8))
        XCTAssertNoThrow(try config.validate())
        XCTAssertThrowsError(try GLiNERNetwork(config:config,weights:[:]))
        let invalid = try decoder.decode(GLiNERConfig.self,from:Data(json.replacingOccurrences(of:"\"num_attention_heads\":2",with:"\"num_attention_heads\":3").utf8))
        XCTAssertThrowsError(try invalid.validate())
    }
    func testInt8ConfigurationRejectsUnsupportedOrAmbiguousFormats() throws {
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let valid = #"{"bits":8,"group_size":64,"mode":"affine","quantized_keys":["encoder.embeddings.word_embeddings.weight"]}"#
        XCTAssertNoThrow(try decoder.decode(GLiNERQuantization.self,from:Data(valid.utf8)).validate())
        for json in [valid.replacingOccurrences(of:"\"bits\":8",with:"\"bits\":4"),
                     valid.replacingOccurrences(of:"\"group_size\":64",with:"\"group_size\":32"),
                     valid.replacingOccurrences(of:"encoder.embeddings.word_embeddings.weight",with:"classifier.layers.0.weight")] {
            XCTAssertThrowsError(try decoder.decode(GLiNERQuantization.self,from:Data(json.utf8)).validate())
        }
    }
    /// Tiny deterministic network: 2 layers, 16 hidden, 2 heads, 8 buckets.
    private func tinyNetwork() throws -> GLiNERNetwork {
        let json = #"{"hidden_size":16,"num_hidden_layers":2,"num_attention_heads":2,"intermediate_size":32,"vocab_size":40,"position_buckets":8,"max_position_embeddings":32,"layer_norm_eps":0.0000001,"relative_attention":true,"share_att_key":true,"position_biased_input":false,"pos_att_type":["p2c","c2p"],"norm_rel_ebd":"layer_norm","type_vocab_size":0}"#
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(GLiNERConfig.self, from: Data(json.utf8))
        var state: UInt64 = 0x9E3779B97F4A7C15
        func values(_ count: Int) -> [Float] {
            (0..<count).map { _ in
                state = state &* 6364136223846793005 &+ 1442695040888963407
                return Float(Int(state >> 40) % 2001 - 1000) / 4000
            }
        }
        var weights = [String: MLXArray]()
        let h = 16, i = 32
        func add(_ key: String, _ shape: [Int]) { weights[key] = MLXArray(values(shape.reduce(1, *)), shape) }
        func linear(_ key: String, _ input: Int, _ output: Int) { add(key + ".weight", [output, input]); add(key + ".bias", [output]) }
        func norm(_ key: String) { weights[key + ".weight"] = MLXArray(values(h).map { 1 + $0 }, [h]); add(key + ".bias", [h]) }
        add("encoder.embeddings.word_embeddings.weight", [40, h]); add("encoder.encoder.rel_embeddings.weight", [16, h])
        norm("encoder.embeddings.LayerNorm"); norm("encoder.encoder.LayerNorm")
        for layer in 0..<2 {
            let p = "encoder.encoder.layers.\(layer)"
            for name in ["query_proj", "key_proj", "value_proj"] { linear(p + ".attention.self_attn." + name, h, h) }
            linear(p + ".attention.output.dense", h, h); norm(p + ".attention.output.LayerNorm")
            linear(p + ".intermediate.dense", h, i); linear(p + ".output.dense", i, h); norm(p + ".output.LayerNorm")
        }
        linear("classifier.layers.0", h, 2 * h); linear("classifier.layers.2", 2 * h, 1)
        linear("count_pred.layers.0", h, 2 * h); linear("count_pred.layers.2", 2 * h, 20)
        add("count_embed.pos_embedding.weight", [20, h])
        for k in ["ih", "hh"] { add("count_embed.gru.weight_\(k)_l0", [3 * h, h]); add("count_embed.gru.bias_\(k)_l0", [3 * h]) }
        linear("count_embed.projector.layers.0", 2 * h, 4 * h); linear("count_embed.projector.layers.2", 4 * h, h)
        for p in ["project_start", "project_end", "out_project"] {
            let key = "span_rep.span_rep_layer.\(p).layers"
            linear(key + ".0", p == "out_project" ? 2 * h : h, 4 * h); linear(key + ".3", 4 * h, h)
        }
        return try GLiNERNetwork(config: config, weights: weights)
    }

    /// The original per-request formulation: project every relative-position
    /// row in every layer and gather from the full bucket range.
    private func referenceEncode(_ net: GLiNERNetwork, _ ids: [Int]) -> MLXArray {
        let c = net.config, n = ids.count, heads = c.numAttentionHeads, d = c.hiddenSize / heads
        var x = net.norm(net.embedding(ids), "encoder.embeddings.LayerNorm")
        let relative = net.norm(net.weights["encoder.encoder.rel_embeddings.weight"]!, "encoder.encoder.LayerNorm")
        let ci = broadcast(MLXArray(GLiNERNetwork.relativeIndices(length: n, buckets: c.positionBuckets, maxPosition: c.maxPositionEmbeddings)).reshaped(1,n,n), to: [heads,n,n])
        let pi = broadcast(MLXArray(GLiNERNetwork.relativeIndices(length: n, buckets: c.positionBuckets, maxPosition: c.maxPositionEmbeddings, reverse: true)).reshaped(1,n,n), to: [heads,n,n])
        let scale = Float(1 / sqrt(Double(d * 3)))
        for i in 0..<c.numHiddenLayers {
            let p = "encoder.encoder.layers.\(i)", a = p + ".attention.self_attn"
            let q = net.heads(net.linear(x, a + ".query_proj")), k = net.heads(net.linear(x, a + ".key_proj")), v = net.heads(net.linear(x, a + ".value_proj"))
            let pq = net.heads(net.linear(relative, a + ".query_proj")), pk = net.heads(net.linear(relative, a + ".key_proj"))
            let c2p = takeAlong(matmul(q, pk.transposed(0,2,1)), ci, axis: -1)
            let p2c = takeAlong(matmul(k, pq.transposed(0,2,1)), pi, axis: -1).transposed(0,2,1)
            let scores = (matmul(q, k.transposed(0,2,1)) + c2p + p2c) * scale
            let context = matmul(softmax(scores, axis: -1), v).transposed(1,0,2).reshaped(n, c.hiddenSize)
            x = net.norm(x + net.linear(context, p + ".attention.output.dense"), p + ".attention.output.LayerNorm")
            x = net.norm(x + net.linear(gelu(net.linear(x, p + ".intermediate.dense")), p + ".output.dense"), p + ".output.LayerNorm")
        }
        return x
    }

    /// Cached position projections and the narrowed bucket band must not
    /// change the encoder output, for short inputs (linear buckets only) and
    /// inputs long enough to reach the log-bucketed range.
    func testCachedPositionProjectionsMatchReference() throws {
        let net = try tinyNetwork()
        for length in [1, 3, 7, 30] {
            let ids = (0..<length).map { ($0 * 7 + 3) % 40 }
            let delta = abs(net.encode(ids) - referenceEncode(net, ids)).max().item(Float.self)
            XCTAssertLessThan(delta, 1e-5, "length \(length)")
        }
    }

    func testPublishedVariantsRequestTheirFiles() {
        XCTAssertEqual(GLiNER.Variant.allCases.map(\.modelID), [
            "aufklarer/GLiNER2.5-Decide-340M-MLX",
            "aufklarer/GLiNER2.5-Decide-340M-MLX-fp16",
            "aufklarer/GLiNER2.5-Decide-340M-MLX-8bit",
        ])
        XCTAssertTrue(GLiNER.Variant.int8.files.contains("quantization.json"))
        XCTAssertFalse(GLiNER.Variant.fp16.files.contains("quantization.json"))
        for variant in GLiNER.Variant.allCases {
            XCTAssertTrue(variant.files.contains("weights.safetensors"))
            XCTAssertTrue(variant.files.contains("encoder_config/config.json"))
        }
    }

    func testRelativeBuckets() {
        XCTAssertEqual(GLiNERNetwork.relativeIndices(length:3,buckets:256,maxPosition:512),[256,255,254,257,256,255,258,257,256])
        XCTAssertEqual(GLiNERNetwork.relativeIndices(length:3,buckets:256,maxPosition:512,reverse:true),[256,257,258,255,256,257,254,255,256])
    }
}
final class E2EGLiNERTests: XCTestCase {
    func testRealCheckpoint() async throws {
        guard let path = ProcessInfo.processInfo.environment["GLINER_MODEL_DIR"] else { throw XCTSkip("Set GLINER_MODEL_DIR to the exported checkpoint") }
        let tolerance: Float = ProcessInfo.processInfo.environment["GLINER_SCORE_TOLERANCE"].flatMap(Float.init) ?? 1e-3
        let model = try await GLiNER.load(from: URL(fileURLWithPath:path))
        let result = try model.classify("Remind me to call Dad at six PM.",labels:["create_reminder","create_calendar_event","send_message","search_notes","set_timer","other"])
        XCTAssertEqual(result.max(by: { $0.probability < $1.probability })?.label,"create_reminder")
        XCTAssertEqual(result.reduce(0) { $0 + $1.probability },1,accuracy:1e-4)
        let entities = try model.extractEntities("Remind me to call Dad at six PM.",labels:["person","time"],descriptions:["person":"Person name or family member mentioned in the command","time":"Time of day or duration mentioned in the command"])
        XCTAssertEqual(entities["person"]?.map(\.text),["Dad"])
        XCTAssertEqual(entities["time"]?.map(\.text),["six PM"])
        guard let fixturePath = ProcessInfo.processInfo.environment["GLINER_REFERENCE_FILE"] else { return }
        let rows = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:fixturePath))) as? [[String:Any]])
        let labels = ["create_reminder","create_calendar_event","send_message","search_notes","set_timer","other"]
        let descriptions = ["person":"Person name or family member mentioned in the command","time":"Time of day or duration mentioned in the command"]
        for row in rows {
            let text = try XCTUnwrap(row["text"] as? String)
            let routing = row["task"] as? String == "routing"
            let prepared = try model.prepare(text:text,task:routing ? "action" : "entities",labels:routing ? labels : ["person","time"],descriptions:routing ? [:] : descriptions,marker:routing ? "[L]" : "[E]")
            XCTAssertEqual(prepared.ids,row["input_ids"] as? [Int],"Tokenizer parity: \(text)")
            let output = try XCTUnwrap(row["output"] as? [String:Any])
            if routing {
                let gold = try XCTUnwrap(output["action"] as? [String:Any])
                let choices = try model.classify(text,labels:labels)
                let best = try XCTUnwrap(choices.max(by: { $0.probability < $1.probability }))
                XCTAssertEqual(best.label,gold["label"] as? String,text)
                XCTAssertEqual(best.probability,try XCTUnwrap((gold["confidence"] as? NSNumber)?.floatValue),accuracy:tolerance,text)
            } else {
                let gold = try XCTUnwrap(output["entities"] as? [String:[[String:Any]]])
                let spans = try model.extractEntities(text,labels:["person","time"],descriptions:descriptions)
                for label in ["person","time"] {
                    let expected = gold[label] ?? []
                    XCTAssertEqual((spans[label] ?? []).map(\.text).sorted(),expected.compactMap { $0["text"] as? String }.sorted(),text)
                    for span in spans[label] ?? [] {
                        let match = try XCTUnwrap(expected.first { $0["text"] as? String == span.text })
                        XCTAssertEqual(span.score,try XCTUnwrap((match["confidence"] as? NSNumber)?.floatValue),accuracy:tolerance,text)
                    }
                }
            }
        }
    }
}
