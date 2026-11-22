"""Explicit local MLX tests, including small byte-range samples of the pinned model.

Run on Apple Silicon: python scripts/tests/e2e_inkling_export.py
Downloads only the first expert from each source precision (about 8 MB).
No complete checkpoint is downloaded and no text generation is performed.
"""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import urllib.request

import mlx.core as mx
import numpy as np

spec = importlib.util.spec_from_file_location(
    "export_inkling", Path(__file__).parents[1] / "export_inkling_mlx.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fetch_range(url, start, length):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise ValueError("Server did not honor the bounded byte-range request")
        result = response.read(length + 1)
        if len(result) != length:
            raise ValueError("Truncated or oversized byte-range response")
        return result


class E2EInklingExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.set_default_device(mx.cpu)
        mx.set_cache_limit(32 * 1024 * 1024)
        cls.url = (f"https://huggingface.co/{exporter.SOURCE_MODEL}/resolve/"
                   f"{exporter.SOURCE_REVISION}/model-00001-of-00001.safetensors")
        size = struct.unpack("<Q", fetch_range(cls.url, 0, 8))[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("Unexpected header size")
        cls.header = json.loads(fetch_range(cls.url, 8, size))
        cls.start = 8 + size
        cls.samples = {}
        for layer, transform in ((0, "nvfp4_nearest"), (20, "mxfp8_nearest")):
            prefix = f"model.llm.layers.{layer}.mlp.experts.w13_weight"
            raw = cls.sample(prefix)
            scales = cls.sample(prefix + "_scale")
            outer = 1.
            if layer == 0:
                outer = cls.sample(prefix + "_amax").item() / cls.sample(prefix + "_outer_scale_denominator").item()
            cls.samples[transform] = (raw, scales, outer)

    @classmethod
    def sample(cls, name):
        info = cls.header[name]
        shape = info["shape"]
        # Take expert zero, or the scalar denominator. Each sample is contiguous.
        sample_shape = shape[1:] if len(shape) == 3 else [1]
        dtype = np.dtype(exporter.RAW_DTYPES[info["dtype"]])
        length = int(np.prod(sample_shape)) * dtype.itemsize
        data = fetch_range(cls.url, cls.start + info["data_offsets"][0], length)
        return np.frombuffer(data, dtype=dtype).reshape(sample_shape)

    def test_source_decoding_matches_native_mlx_quantization_formats(self):
        for transform, (raw, scales, outer) in self.samples.items():
            with self.subTest(transform=transform):
                expected = exporter.dequantize_expert(raw, scales, transform, outer)
                packed = mx.array(np.ascontiguousarray(raw)).view(mx.uint32)
                mode, group, bits = ("nvfp4", 16, 4) if transform == "nvfp4_nearest" else ("mxfp8", 32, 8)
                native = mx.dequantize(packed, mx.array(scales), group_size=group, bits=bits,
                                       mode=mode, dtype=mx.float32)
                native = native * outer
                np.testing.assert_allclose(np.array(native), expected, rtol=1e-6, atol=1e-7)

    def test_real_expert_affine_export_roundtrip_and_matrix_product(self):
        rng = np.random.default_rng(7)
        for transform, (raw, scales, outer) in self.samples.items():
            # The fused gate/up rows alternate in the source, rather than forming halves.
            dense = exporter.dequantize_expert(raw, scales, transform, outer)
            for bits, tolerance in ((4, .15), (5, .08), (8, .02)):
                for label, part in (("gate", dense[0::2]), ("up", dense[1::2])):
                    with self.subTest(transform=transform, bits=bits, projection=label):
                        value = mx.array(np.ascontiguousarray(part)).astype(mx.bfloat16)
                        packed, s, b = mx.quantize(value, group_size=64, bits=bits)
                        # Match the FP32 product's arithmetic; rounding a dequantized
                        # matrix to BF16 first introduces an unrelated second error.
                        restored = mx.dequantize(packed, s.astype(mx.float32), b.astype(mx.float32),
                                                group_size=64, bits=bits)
                        actual = np.array(restored.astype(mx.float32))
                        nrmse = np.linalg.norm(actual - part) / np.linalg.norm(part)
                        self.assertLess(nrmse, tolerance)
                        x = mx.array(rng.standard_normal((2, part.shape[1])).astype(np.float32))
                        product = mx.quantized_matmul(x, packed, s, b, transpose=True, group_size=64, bits=bits)
                        np.testing.assert_allclose(np.array(product), np.array(x) @ actual.T,
                                                   rtol=1e-4, atol=1e-4)
                mx.clear_cache()

    def test_shards_reload_with_complete_index_and_exact_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = mx.array([[1., 2.], [3., 4.]]).astype(mx.bfloat16)
            writer = exporter.ShardWriter(root, max_bytes=8)
            writer.add("first.weight", value)
            writer.add("second.weight", value)
            index = writer.finish()
            self.assertEqual(len(set(index["weight_map"].values())), 2)
            self.assertEqual(index["metadata"]["total_size"], 16)
            for name, filename in index["weight_map"].items():
                loaded = mx.load(str(root / filename))
                self.assertEqual(loaded[name].dtype, mx.bfloat16)
                np.testing.assert_array_equal(np.array(loaded[name].astype(mx.float32)), [[1, 2], [3, 4]])

    def test_expert_export_preserves_interleaved_gate_up_order(self):
        # Small real safetensors with the same 128-expert packing as the source.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = "model.llm.layers.0.mlp.experts.w13_weight"
            raw = np.full((128, 64, 32), 0x22, dtype=np.uint8)
            raw[:, 1::2] = 0x44  # gate = 1, up = 2
            tensors = {
                prefix: ("U8", raw),
                prefix + "_scale": ("F8_E4M3", np.full((128, 64, 4), 56, dtype=np.uint8)),
                prefix + "_amax": ("F32", np.ones((128,), dtype=np.float32)),
                prefix + "_outer_scale_denominator": ("F32", np.ones((1,), dtype=np.float32)),
            }
            header, payload = {}, bytearray()
            for name, (dtype, data) in tensors.items():
                start = len(payload)
                payload.extend(data.tobytes())
                header[name] = {"dtype": dtype, "shape": list(data.shape), "data_offsets": [start, len(payload)]}
            encoded = json.dumps(header).encode()
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            exporter.write_json(root / "model.safetensors.index.json", {
                "weight_map": {n: "model.safetensors" for n in tensors}})
            source = exporter.TensorSource(root)
            weights = dict(exporter.export_experts(source, prefix, "nvfp4_nearest", 4, 64))
            base = "language_model.model.layers.0.mlp.switch_mlp."
            for projection, expected in (("gate_proj", 1.), ("up_proj", 2.)):
                p = base + projection
                result = mx.dequantize(weights[p + ".weight"], weights[p + ".scales"],
                                       weights[p + ".biases"], group_size=64, bits=4)
                self.assertEqual(result.shape, (128, 32, 64))
                np.testing.assert_allclose(np.array(result.astype(mx.float32)), expected, atol=.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
