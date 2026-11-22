"""CPU-only exporter regression tests; no MLX import or model download."""

import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import io

import numpy as np

spec = importlib.util.spec_from_file_location(
    "export_inkling", Path(__file__).parents[1] / "export_inkling_mlx.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def config():
    return {"model_type": "inkling2_loop", "quantization_config": {"quant_method": "modelopt"},
            "text_config": {"model_type": "inkling2_loop_text", "hidden_size": 2304,
                            "num_hidden_layers": 24, "n_routed_experts": 128,
                            "intermediate_size": 1152, "num_experts_per_tok": 8,
                            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3,
                                            "gate_rank": 64, "is_varloop": False}}}


def write_tensor_file(root, *, dtype="BF16", shape=(2, 2), data=None, offsets=None, filename="model.safetensors"):
    if data is None:
        data = np.array([[0x3F80, 0x4000], [0x4040, 0x4080]], dtype="<u2").tobytes()
    header = {"test.weight": {"dtype": dtype, "shape": list(shape),
                              "data_offsets": offsets or [0, len(data)]}}
    encoded = json.dumps(header).encode()
    (root / filename).write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"test.weight": filename}}))


class InklingExportConfigTests(unittest.TestCase):
    def test_invalid_output_is_rejected_before_any_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep.txt").write_text("existing output")
            with patch.object(exporter, "resolve_source") as download:
                with self.assertRaisesRegex(ValueError, "empty"):
                    exporter.main(["--output", str(root)])
                download.assert_not_called()
    def test_source_revision_is_pinned_and_rejects_mutable_revisions(self):
        self.assertRegex(exporter.SOURCE_REVISION, r"^[a-f0-9]{40}$")
        with self.assertRaises(SystemExit):
            exporter.parse_args(["--output", "out", "--source-revision", "main"])

    def test_precision_and_shard_arguments(self):
        args = exporter.parse_args(["--output", "out", "--bits", "5", "--include-mtp"])
        self.assertEqual((args.bits, args.group_size, args.include_mtp), (5, 64, True))
        for arguments in (["--bits", "3"], ["--max-shard-size-mb", "0"]):
            with self.assertRaises(SystemExit):
                exporter.parse_args(["--output", "out", *arguments])

    def test_requires_looped_mini_architecture(self):
        exporter.validate_config(config())
        for field, value in (("hidden_size", 4608), ("n_routed_experts", 384)):
            invalid = config()
            invalid["text_config"][field] = value
            with self.assertRaisesRegex(ValueError, "26B"):
                exporter.validate_config(invalid)
        invalid = config()
        invalid["text_config"]["loop_config"]["num_passes"] = 1
        with self.assertRaisesRegex(ValueError, "loop_config"):
            exporter.validate_config(invalid)

    def test_rejects_already_converted_checkpoint(self):
        invalid = config()
        invalid["quantization_config"] = {"bits": 4, "group_size": 64}
        with self.assertRaisesRegex(ValueError, "upstream"):
            exporter.validate_config(invalid)

    def test_quantization_metadata_covers_all_parameters(self):
        groups = [{"weight_transform": "cast", "tensors": ["norm.weight"]},
                  {"weight_transform": "nvfp4_nearest", "tensors": ["expert"]}]
        metadata = {"quant_method": "inkling2", "model": {"parameter_groups": groups}}
        names = {"norm.weight", "expert", "expert_scale", "expert_amax", "expert_outer_scale_denominator"}
        self.assertEqual(exporter.weight_transforms(metadata, names)["expert"], "nvfp4_nearest")
        for bad_names in (names | {"unclassified.weight"}, names - {"norm.weight"}):
            with self.assertRaises(ValueError):
                exporter.weight_transforms(metadata, bad_names)
        duplicated = copy.deepcopy(metadata)
        duplicated["model"]["parameter_groups"].append(groups[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            exporter.weight_transforms(duplicated, names)


class InklingExportNumericsTests(unittest.TestCase):
    def test_e4m3_subnormals_signed_zero_extrema_and_nan(self):
        raw = np.array([0, 128, 1, 8, 56, 64, 126, 254, 127, 255], dtype=np.uint8)
        decoded = exporter.fp8_e4m3(raw)
        np.testing.assert_array_equal(decoded[:8], [0, -0., 2**-9, 2**-6, 1, 2, 448, -448])
        self.assertTrue(np.signbit(decoded[1]))
        self.assertTrue(np.isnan(decoded[8:]).all())

    def test_nvfp4_nibble_order_and_outer_scale(self):
        # Each byte contains the earlier value in its low nibble. Scale 0x38 is 1.0.
        packed = np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=np.uint8)
        actual = exporter.dequantize_expert(packed, np.array([[56]], dtype=np.uint8),
                                            "nvfp4_nearest", 2.)
        np.testing.assert_array_equal(actual, exporter.FP4_VALUES[np.arange(16)][None, :] * 2)

    def test_nvfp4_scales_apply_to_groups_not_whole_rows(self):
        packed = np.full((2, 16), 0x22, dtype=np.uint8)
        scales = np.array([[56, 64], [64, 56]], dtype=np.uint8)
        actual = exporter.dequantize_expert(packed, scales, "nvfp4_nearest")
        np.testing.assert_array_equal(actual[0], np.repeat([1, 2], 16))
        np.testing.assert_array_equal(actual[1], np.repeat([2, 1], 16))

    def test_mxfp8_uses_ue8m0_scales_in_groups_of_32(self):
        raw = np.full((2, 64), 56, dtype=np.uint8)  # FP8 1.0
        scales = np.array([[127, 128], [126, 129]], dtype=np.uint8)
        actual = exporter.dequantize_expert(raw, scales, "mxfp8_nearest")
        np.testing.assert_array_equal(actual[0], np.repeat([1, 2], 32))
        np.testing.assert_array_equal(actual[1], np.repeat([.5, 4], 32))

    def test_rejects_invalid_scale_shapes_and_nonfinite_values(self):
        raw = np.zeros((1, 32), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "shape"):
            exporter.dequantize_expert(raw, np.zeros((1, 2), dtype=np.uint8), "mxfp8_nearest")
        with self.assertRaisesRegex(ValueError, "NaN"):
            exporter.dequantize_expert(raw, np.array([[255]], dtype=np.uint8), "mxfp8_nearest")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            exporter.dequantize_expert(np.full((1, 32), 127, dtype=np.uint8),
                                       np.array([[127]], dtype=np.uint8), "mxfp8_nearest")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            exporter.dequantize_expert(raw, raw, "unknown")


class InklingExportLayoutTests(unittest.TestCase):
    def test_output_size_accounts_for_packing_and_affine_auxiliaries(self):
        shapes = {
            "model.llm.layers.0.mlp.experts.w13_weight": {"shape": [2, 64, 32], "dtype": "U8"},
            "model.llm.embed.weight": {"shape": [64, 64], "dtype": "BF16"},
            "model.llm.norm.weight": {"shape": [64], "dtype": "F32"},
        }
        source = type("HeaderSource", (), {"info": lambda _, name: shapes[name]})()
        transforms = {name: "nvfp4_nearest" if "experts" in name else "cast" for name in shapes}
        self.assertEqual(exporter.estimated_weight_bytes(source, transforms, 4, 32), 5120 + 4608 + 256)
    def test_maps_attention_norms_and_loop_controllers(self):
        cases = {
            "model.llm.layers.6.attn.wq_du.weight": "language_model.model.layers.6.self_attn.q_proj.weight",
            "model.llm.layers.6.attn.q_l2_clip.scale": "language_model.model.layers.6.self_attn.q_scale",
            "model.llm.layers.6.attn.rel_logits_proj.proj": "language_model.model.layers.6.self_attn.rel_proj",
            "model.llm.layers.11.attn.k_sconv.static_w": "language_model.model.layers.11.self_attn.k_sconv.static_weight",
            "model.llm.layers.11.attn.k_sconv.weight_proj": "language_model.model.layers.11.self_attn.k_sconv.weight_proj.weight",
            "model.llm.layers.6.controllers.attn.down_weight": "language_model.model.layers.6.controllers.attn.down_weight",
            "model.llm.layers.11.q_input_norm.weight": "language_model.model.layers.11.q_input_layernorm.weight",
            "model.llm.layers.6.mlp.gate.bias": "language_model.model.layers.6.mlp.e_score_correction_bias",
            "model.llm.attn_res.attn_queries.5": "language_model.model.layers.6.attention_input.query",
            "model.llm.attn_res.mlp_queries.5": "language_model.model.layers.5.mlp_input.query",
            "model.llm.attn_res.output_query": "language_model.model.final_input.query",
            "model.llm.embed.weight": "language_model.model.embed_tokens.weight",
            "model.llm.unembed.weight": "language_model.lm_head.weight",
            "model.visual.backbone.layers.0.ln_1.weight": "vision_tower.layers.0.layernorm_before.weight",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(exporter.mapped_name(name), (expected, False))

    def test_convolution_transposes_only_static_conv1d_weights(self):
        for sub in ("attn.k_sconv.weight", "attn.v_sconv.weight", "attn_sconv.weight", "mlp_sconv.weight"):
            target, conv = exporter.mapped_name("model.llm.layers.0." + sub)
            self.assertTrue(conv)
            self.assertTrue(target.endswith(".conv.weight"))

    def test_preserves_sensitive_matrices(self):
        for name in ("vision_tower.layers.0.mlp.fc1.weight", "language_model.model.layers.6.mlp.gate_weight",
                     "vision_tower.layers.0.self_attn.q_proj.weight",
                     "vision_tower.layers.0.self_attn.k_proj.weight",
                     "vision_tower.layers.0.self_attn.v_proj.weight",
                     "language_model.model.layers.6.controllers.attn.down_weight",
                     "language_model.model.layers.0.self_attn.rel_proj",
                     "language_model.model.layers.5.self_attn.k_sconv.weight_proj.weight"):
            self.assertFalse(exporter.quantize_dense(name, (128, 128)))
        self.assertTrue(exporter.quantize_dense("language_model.lm_head.weight", (201024, 2304)))
        self.assertTrue(exporter.quantize_dense("language_model.model.layers.0.self_attn.q_proj.weight", (3072, 2304)))


class InklingTensorSourceTests(unittest.TestCase):
    def test_remote_ranges_reject_wrong_or_unbounded_responses(self):
        source = exporter.RemoteTensorSource.__new__(exporter.RemoteTensorSource)
        source.revision = exporter.SOURCE_REVISION
        for status, content_range, data in [(200, "bytes 8-11/20", b"test"),
                                            (206, "bytes 0-3/20", b"test"),
                                            (206, "bytes 8-11/20", b"too long")]:
            response = io.BytesIO(data)
            response.status = status
            response.headers = {"Content-Range": content_range}
            with patch.object(exporter.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    source.fetch("model.safetensors", 8, 4)
    def test_maps_bf16_bits_without_converting_or_loading_whole_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensor_file(root)
            source = exporter.TensorSource(root)
            self.assertEqual(source.info("test.weight")["dtype"], "BF16")
            self.assertIsInstance(source.raw("test.weight"), np.memmap)
            np.testing.assert_array_equal(source.raw("test.weight"), [[0x3F80, 0x4000], [0x4040, 0x4080]])

    def test_rejects_truncated_source_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensor_file(root, offsets=[0, 80])
            with self.assertRaisesRegex(ValueError, "offsets"):
                exporter.TensorSource(root)

    def test_rejects_index_pointing_outside_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"test.weight": "../outside.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "filename"):
                exporter.TensorSource(root)

    def test_rejects_index_header_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tensor_file(root)
            index = root / "model.safetensors.index.json"
            index.write_text(index.read_text().replace("test.weight", "wrong.weight"))
            with self.assertRaisesRegex(ValueError, "index"):
                exporter.TensorSource(root)


if __name__ == "__main__":
    unittest.main()
