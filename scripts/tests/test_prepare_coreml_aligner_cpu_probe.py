"""Small graph fixtures and mocked downloads; no Core ML inference or model files."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from coremltools.proto import MIL_pb2 as mil, Model_pb2
from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType

MODULE = Path(__file__).resolve().parents[1] / "prepare_coreml_aligner_cpu_probe.py"
SPEC = importlib.util.spec_from_file_location("cpu_probe", MODULE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def graph_fixture():
    spec = Model_pb2.Model()
    spec.specificationVersion = 8
    output = spec.description.output.add(name="logits")
    output.type.multiArrayType.dataType = ArrayFeatureType.FLOAT16
    function = spec.mlProgram.functions["main"]
    function.inputs.add(name="inputs_embeds").type.tensorType.dataType = mil.FLOAT32
    block = function.block_specializations["CoreML7"]
    block.outputs.append("logits")

    def operation(kind, name, output_dtype, **inputs):
        op = block.operations.add(type=kind)
        op.outputs.add(name=name).type.tensorType.dataType = output_dtype
        for key, value in inputs.items():
            op.inputs[key].arguments.add(name=value)
        return op

    def constant(name, dtype, value):
        op = operation("const", name, dtype)
        tensor = op.attributes["val"]
        tensor.type.tensorType.dataType = dtype
        if dtype == mil.STRING:
            tensor.immediateValue.tensor.strings.values.append(value)
        elif dtype == mil.FLOAT16:
            tensor.immediateValue.tensor.bytes.values = np.array([value], dtype="<f2").tobytes()
        else:
            tensor.immediateValue.tensor.floats.values.append(value)

    constant("dtype16", mil.STRING, "fp16")
    constant("two", mil.FLOAT16, 2)
    constant("epsilon", mil.FLOAT16, 1e-6)
    constant("rsqrt_epsilon", mil.FLOAT32, 1e-12)
    constant("gamma", mil.FLOAT16, 0.5)
    operation("cast", "x", mil.FLOAT16, x="inputs_embeds", dtype="dtype16")
    operation("pow", "square", mil.FLOAT16, x="x", y="two")
    operation("reduce_mean", "mean", mil.FLOAT16, x="square")
    operation("add", "variance", mil.FLOAT16, x="mean", y="epsilon")
    operation("rsqrt", "inverse", mil.FLOAT16, x="variance", epsilon="rsqrt_epsilon")
    operation("mul", "normalized", mil.FLOAT16, x="x", y="inverse")
    operation("mul", "logits", mil.FLOAT16, x="normalized", y="gamma")
    return spec


def operations(spec):
    block = spec.mlProgram.functions["main"].block_specializations["CoreML7"]
    return {op.outputs[0].name: op for op in block.operations}


def evaluate(spec, x, pow_fault=False):
    """Run the fixture graph with numpy in FP16.

    With pow_fault, pow(x, 2) returns 1 for a negative x, as measured on the
    Neural Engine of an M1 Pro. The CPU and the GPU compute it correctly.
    """
    block = spec.mlProgram.functions["main"].block_specializations["CoreML7"]
    math = {
        "cast": lambda a: a["x"].astype(np.float16),
        "pow": lambda a: (np.where(a["x"] < 0, np.float16(1), a["x"] ** a["y"]) if pow_fault
                          else a["x"] ** a["y"]),
        "reduce_mean": lambda a: a["x"].mean(axis=-1, keepdims=True, dtype=np.float16),
        "add": lambda a: a["x"] + a["y"],
        "rsqrt": lambda a: np.float16(1) / np.sqrt(a["x"]),
        "mul": lambda a: a["x"] * a["y"],
    }
    values = {"inputs_embeds": x}
    for op in block.operations:
        if op.type == "const":
            tensor = op.attributes["val"].immediateValue.tensor
            kind = tensor.WhichOneof("value")
            if kind == "bytes":
                value = np.frombuffer(tensor.bytes.values, dtype="<f2")[0]
            else:
                value = getattr(tensor, kind).values[0]
                if kind == "floats":
                    value = np.float16(value)
        else:
            value = math[op.type]({key: values[probe.reference(op, key)] for key in op.inputs})
        values[op.outputs[0].name] = value
    return values["logits"]


class GraphCandidateTests(unittest.TestCase):
    def setUp(self):
        self.source = graph_fixture()
        self.original = self.source.SerializeToString()

    def tearDown(self):
        self.assertEqual(self.source.SerializeToString(), self.original)

    def test_recompile_is_an_unchanged_control(self):
        result = probe.transform(self.source, "recompile")
        self.assertEqual(result.SerializeToString(), self.original)
        self.assertIsNot(result, self.source)

    def test_output32_preserves_the_external_name_and_internal_math(self):
        result = probe.transform(self.source, "output32")
        changed = operations(result)
        self.assertEqual(result.description.output[0].name, "logits")
        self.assertEqual(result.description.output[0].type.multiArrayType.dataType,
                         ArrayFeatureType.FLOAT32)
        self.assertEqual(changed["logits"].type, "cast")
        self.assertEqual(probe.reference(changed["logits"], "x"), "cpu_compat_logits_fp16")
        self.assertEqual(changed["cpu_compat_logits_fp16"].outputs[0].type.tensorType.dataType,
                         mil.FLOAT16)
        for name, op in operations(self.source).items():
            if name != "logits":
                self.assertEqual(changed[name], op)

    def test_norm32_promotes_math_and_restores_the_original_weighted_interface(self):
        result = probe.transform(self.source, "norm32", expected_norms=1)
        changed = operations(result)
        self.assertEqual(result.description, self.source.description)
        self.assertEqual(changed["gamma"], operations(self.source)["gamma"])
        self.assertEqual(changed["logits"], operations(self.source)["logits"])
        for name in ("square", "mean", "variance", "inverse", "cpu_norm_0_normalized"):
            self.assertEqual(changed[name].outputs[0].type.tensorType.dataType, mil.FLOAT32)
        self.assertEqual(changed["normalized"].type, "cast")
        self.assertEqual(changed["normalized"].outputs[0].type.tensorType.dataType, mil.FLOAT16)
        epsilon = changed["cpu_norm_0_epsilon"].attributes["val"].immediateValue.tensor.floats.values[0]
        self.assertEqual(epsilon, float(np.float16(1e-6)))
        self.assertEqual(probe.reference(changed["cpu_norm_0_normalized"], "x"), "cpu_norm_0_input")

    def test_unrecognized_normalization_fails_closed(self):
        modified = graph_fixture()
        operations(modified)["variance"].type = "sub"
        with self.assertRaisesRegex(ValueError, "Unrecognized RMS"):
            probe.transform(modified, "norm32", expected_norms=1)

    def test_unexpected_chain_count_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Expected 113 normalization chains, found 1"):
            probe.transform(self.source, "norm32")

    def test_unexpected_output_interface_fails_closed(self):
        modified = graph_fixture()
        modified.description.output[0].name = "renamed"
        with self.assertRaisesRegex(ValueError, "FP16 logits"):
            probe.transform(modified, "output32")

    def test_nested_blocks_are_rejected(self):
        modified = graph_fixture()
        operations(modified)["square"].blocks.add()
        with self.assertRaisesRegex(ValueError, "Nested"):
            probe.transform(modified, "output32")

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown candidate"):
            probe.transform(self.source, "invalid")

    def test_mulsquare_writes_the_square_as_a_product_and_adds_nothing(self):
        result = probe.transform(self.source, "mulsquare", expected_norms=1)
        changed, source = operations(result), operations(self.source)
        self.assertEqual(result.description, self.source.description)
        self.assertEqual(set(changed), set(source))
        self.assertEqual(changed["square"].type, "mul")
        self.assertEqual([probe.reference(changed["square"], key) for key in ("x", "y")], ["x", "x"])
        self.assertEqual(changed["square"].outputs[0], source["square"].outputs[0])
        for name in set(source) - {"square"}:
            self.assertEqual(changed[name], source[name])

    def test_mulsquare_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Expected 113 normalization chains, found 1"):
            probe.transform(self.source, "mulsquare")
        modified = graph_fixture()
        operations(modified)["variance"].type = "sub"
        with self.assertRaisesRegex(ValueError, "Unrecognized RMS"):
            probe.transform(modified, "mulsquare", expected_norms=1)
        modified = graph_fixture()
        exponent = operations(modified)["two"].attributes["val"].immediateValue.tensor
        exponent.bytes.values = np.array([3], dtype="<f2").tobytes()
        with self.assertRaisesRegex(ValueError, "Expected a square in the normalization chain"):
            probe.transform(modified, "mulsquare", expected_norms=1)
        modified = graph_fixture()
        operations(modified)["square"].outputs[0].type.tensorType.dataType = mil.FLOAT32
        with self.assertRaisesRegex(ValueError, "Overlapping or non-FP16 normalization chain"):
            probe.transform(modified, "mulsquare", expected_norms=1)

    def test_mulsquare_is_not_a_nightly_candidate(self):
        self.assertEqual(probe.MODES, ("recompile", "output32", "norm32"))

    def test_mulsquare_is_right_where_pow_returns_one_for_negative_inputs(self):
        rows = np.random.default_rng(0).standard_normal((2, 1024)).astype(np.float32) * 0.02
        rows[1] = 0
        wide = rows.astype(np.float64)
        expected = wide / np.sqrt((wide * wide).mean(-1, keepdims=True) + 1e-6) * 0.5

        def close(spec):
            return np.allclose(evaluate(spec, rows, pow_fault=True), expected, rtol=0.01, atol=1e-3)

        self.assertFalse(close(self.source))  # Control: the published chain is wrong.
        self.assertTrue(close(probe.transform(self.source, "mulsquare", expected_norms=1)))


class PinnedSourceTests(unittest.TestCase):
    def download(self, root, corrupt=False):
        weights = root / "weight.bin"
        weights.write_bytes(b"verified weights")
        weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        files = {"text_decoder.mlpackage/Manifest.json": b"{}",
                 "text_decoder.mlpackage/Data/com.apple.CoreML/model.mlmodel": b"specification"}
        tree = [{"path": path, "type": "file", "size": len(data),
                 "oid": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
                for path, data in files.items()]
        tree[1]["lfs"] = {"oid": hashlib.sha256(files[tree[1]["path"]]).hexdigest()}
        tree.extend([
            {"path": "text_decoder.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
             "type": "file", "lfs": {"oid": weight_hash}},
            {"path": "../../escape", "type": "file", "size": 1, "oid": "invalid"}])
        requested = []

        def fetch(url, timeout):
            requested.append(url)
            if "/api/models/" in url:
                return io.BytesIO(json.dumps(tree).encode())
            prefix = f"https://huggingface.co/{probe.REPO}/resolve/{probe.REVISION}/"
            self.assertTrue(url.startswith(prefix))
            data = files[url.removeprefix(prefix)]
            return io.BytesIO(b"x" * len(data) if corrupt else data)

        package = root / "source.mlpackage"
        with patch.object(probe, "WEIGHT_SHA256", weight_hash), patch.object(probe, "urlopen", fetch):
            probe.download_source(package, weights)
        return package, weights, requested

    def test_verified_weights_are_reused_and_remote_paths_are_allowlisted(self):
        with tempfile.TemporaryDirectory() as directory:
            package, weights, requested = self.download(Path(directory))
            self.assertEqual(len(requested), 3)
            self.assertEqual((package / "Data/com.apple.CoreML/model.mlmodel").read_bytes(), b"specification")
            self.assertEqual((package / "Data/com.apple.CoreML/weights/weight.bin").stat().st_ino,
                             weights.stat().st_ino)

    def test_corrupt_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Invalid source file hash"):
                self.download(Path(directory), corrupt=True)

    def test_unexpected_cached_weights_are_rejected_before_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weight.bin"
            weights.write_bytes(b"wrong")
            with patch.object(probe, "urlopen") as fetch:
                with self.assertRaisesRegex(ValueError, "do not match"):
                    probe.download_source(root / "package", weights)
                fetch.assert_not_called()

    def test_existing_candidates_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                probe.prepare_candidates(Path(directory) / "cache", Path(directory))


if __name__ == "__main__":
    unittest.main()
