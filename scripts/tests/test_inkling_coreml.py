"""CPU-only checks for the Core ML exporter and image preparation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling_coreml import Int8Experts, CompressedLinear, quantize_int8, image_inputs, layer_visits, conversion_inputs
from inkling_coreml_source import Checkpoint, round_bfloat16
from prepare_inkling_coreml_bridge import validate_environment
from export_inkling_coreml import copy_verified_asset


class ResumeMetadataTests(unittest.TestCase):
    def test_readonly_metadata_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            source.write_bytes(b'pinned metadata'); source.chmod(0o444)
            copy_verified_asset(source, target)
            before = target.stat().st_mtime_ns
            copy_verified_asset(source, target)
            self.assertEqual(target.stat().st_mtime_ns, before)
            self.assertEqual(target.read_bytes(), b'pinned metadata')

    def test_mismatched_resume_metadata_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            source.write_bytes(b'pinned'); target.write_bytes(b'different')
            with self.assertRaises(ValueError): copy_verified_asset(source, target)
            self.assertEqual(target.read_bytes(), b'different')


class BridgeSetupTests(unittest.TestCase):
    def test_refuses_system_installation(self):
        with self.assertRaises(ValueError):
            validate_environment('/python', '/python', '/python/site-packages/coremltools')

    def test_refuses_package_outside_venv(self):
        with self.assertRaises(ValueError):
            validate_environment('/venv', '/python', '/python/site-packages/coremltools')

    def test_accepts_package_inside_isolated_venv(self):
        validate_environment('/venv', '/python', '/venv/lib/python3.12/site-packages/coremltools')


class QuantizationTests(unittest.TestCase):
    def test_zero_groups(self):
        values, scales = quantize_int8(torch.zeros(2, 64), 32)
        self.assertEqual(values.dtype, torch.int8)
        self.assertEqual(scales.dtype, torch.float16)
        self.assertEqual(int(torch.count_nonzero(values)), 0)
        self.assertTrue(bool(torch.all(scales > 0)))

    def test_quantization_error_bound(self):
        original = torch.linspace(-2, 2, 256).reshape(4, 64)
        values, scales = quantize_int8(original, 32)
        actual = values.reshape(4, 2, 32).float() * scales.float().unsqueeze(-1)
        error = (actual - original.reshape_as(actual)).abs()
        self.assertTrue(bool(torch.all(error <= scales.float().unsqueeze(-1) * .501)))

    def test_invalid_matrices(self):
        for matrix in [torch.ones(7), torch.ones(2, 63), torch.full((2, 64), float('nan')),
                       torch.full((2, 64), float('inf')), torch.full((2, 64), 1e20)]:
            with self.subTest(shape=matrix.shape):
                with self.assertRaises(ValueError):
                    quantize_int8(matrix)

    def test_selected_experts(self):
        weight = torch.arange(4 * 32 * 64, dtype=torch.float32).reshape(4, 32, 64) / 8000
        module = Int8Experts(weight, 32, torch.float32)
        indices = torch.tensor([[3, 0]], dtype=torch.int32)
        inputs = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64) / 64
        actual = module(inputs, indices)
        selected = module.quantized[[3, 0]].reshape(2, 32, 2, 32).float()
        dense = (selected * module.scales[[3, 0]].float().unsqueeze(-1)).reshape(2, 32, 64)
        expected = torch.matmul(dense, inputs.reshape(64, 1)).squeeze(-1).unsqueeze(0)
        torch.testing.assert_close(actual, expected)

    def test_dense_metadata(self):
        module = CompressedLinear(torch.ones(2, 64), 32)
        buffers = dict(module.named_buffers())
        self.assertEqual(buffers['_COREML_/weight/quantization_n_bits'].item(), 8)
        self.assertEqual(buffers['_COREML_/weight/compression_type'].tolist(), [3])
        self.assertEqual(list(buffers['_COREML_/weight/quantization_scale'].shape), [2, 2])

    def test_half_expert_dequantization_matches_float_reference(self):
        torch.manual_seed(97)
        module = Int8Experts(torch.randn(4, 32, 64) * .1, 32, torch.float16)
        inputs = torch.randn(1, 1, 64).half() * .1
        indices = torch.tensor([[1, 3]], dtype=torch.int32)
        selected = module.quantized[[1, 3]].reshape(2, 32, 2, 32).float()
        dense = (selected * module.scales[[1, 3]].float().unsqueeze(-1)).reshape(2, 32, 64).half()
        expected = torch.matmul(dense, inputs.reshape(64, 1)).squeeze(-1).unsqueeze(0)
        torch.testing.assert_close(module(inputs, indices), expected, atol=0, rtol=0)

    def test_bfloat_rounding(self):
        values = np.random.default_rng(97).normal(size=2048).astype(np.float32)
        expected = torch.from_numpy(values).bfloat16().float().numpy()
        np.testing.assert_array_equal(round_bfloat16(values), expected)

    def test_bfloat_nonfinite(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(ValueError):
                round_bfloat16(np.array([value], dtype=np.float32))


class SourceQueryTests(unittest.TestCase):
    def checkpoint(self, layers=2):
        original = 'model.llm.middle_cycle_cross_pass'
        canonical = 'language_model.model.middle_cycle_cross_pass'
        arrays = {original + '_queries': np.arange(layers * 8, dtype=np.float32).reshape(layers, 2, 4),
                  original + '_kv_queries': 100 + np.arange(layers * 8, dtype=np.float32).reshape(layers, 2, 4)}
        checkpoint = Checkpoint.__new__(Checkpoint)
        checkpoint.config = {'text_config': {'hidden_size': 4, 'loop_config': {'core_start': 1, 'core_end': 3}}}
        checkpoint.names = {canonical + suffix: original + suffix for suffix in ['_queries', '_kv_queries']}
        class Tensors:
            def raw(self, name): return arrays[name]
            def info(self, name): return {'dtype': 'F32'}
        checkpoint.tensors = Tensors()
        return checkpoint

    def test_combines_attention_key_value_mlp_queries_in_order(self):
        checkpoint = self.checkpoint()
        queries = checkpoint.read('language_model.model.message_queries')
        self.assertEqual(queries.shape, (2, 4, 4))
        np.testing.assert_array_equal(queries[0], [[0, 1, 2, 3], [100, 101, 102, 103],
                                                [104, 105, 106, 107], [4, 5, 6, 7]])
        self.assertIs(checkpoint.read('language_model.model.message_queries'), queries)

    def test_rejects_wrong_core_query_count(self):
        with self.assertRaises(ValueError):
            self.checkpoint(layers=3).read('language_model.model.message_queries')


class ImagePreparationTests(unittest.TestCase):
    config = {'native_patch_size': 20, 'adapter_merge_size': 2,
              'hidden_size': 1024, 'num_attention_heads': 16, 'posemb_grid_size': 32}

    def test_patch_order_and_padding(self):
        image = np.zeros((41, 81, 3), dtype=np.uint8)
        image[40, 80] = 255
        result = image_inputs(image, self.config)
        self.assertEqual(result['pixels'].shape, (6, 40, 40, 3))
        np.testing.assert_array_equal(result['pixels'][5, 0, 0], [1, 1, 1])
        np.testing.assert_allclose(result['pixels'][5, 1, 1], [-1/127.5-1] * 3)

    def test_bilinear_coefficients(self):
        result = image_inputs(np.zeros((81, 41, 3), dtype=np.uint8), self.config)
        np.testing.assert_allclose(result['position_coefficients'].sum(-1), 1, atol=1e-6)
        self.assertEqual(result['position_indices'].shape, (24, 4))
        self.assertGreaterEqual(result['position_indices'].min(), 0)
        self.assertLess(result['position_indices'].max(), 1024)

    def test_class_rotary_position(self):
        result = image_inputs(np.zeros((40, 40, 3), dtype=np.uint8), self.config)
        np.testing.assert_array_equal(result['cosine'][0], np.ones(64))
        np.testing.assert_array_equal(result['sine'][0], np.zeros(64))

    def test_image_bounds(self):
        for image in [np.zeros((1, 1), dtype=np.uint8), np.zeros((1, 1, 3), dtype=np.float32),
                      np.zeros((0, 40, 3), dtype=np.uint8), np.zeros((4097, 1, 3), dtype=np.uint8)]:
            with self.assertRaises(ValueError):
                image_inputs(image, self.config)
        with self.assertRaises(ValueError):
            image_inputs(np.zeros((41, 41, 3), dtype=np.uint8), self.config, max_patches=3)


class VisitTests(unittest.TestCase):
    def test_conversion_shapes_can_be_constructed_repeatedly(self):
        # Named RangeDim symbols are process-global. Repeated partition exports
        # must allocate independent symbols instead of re-registering names.
        config = {'hidden_size': 64, 'num_hidden_layers': 8, 'n_layers_per_attn_res_block': 2,
                  'local_layer_ids': [0], 'swa_num_key_value_heads': 2, 'swa_head_dim': 16,
                  'sliding_window_size': 4, 'num_key_value_heads': 2, 'head_dim': 16, 'sconv_kernel_size': 4}
        for _ in range(2):
            specs, sample, _ = conversion_inputs(config, 'layer', context_limit=64, prefill_size=1)
            self.assertEqual(sample['x'].shape, (1, 1, 64))
            self.assertEqual(len(specs), 12)

    def test_shared_layers_have_distinct_visits(self):
        config = {'num_hidden_layers': 8, 'loop_config': {'core_start': 2, 'core_end': 6, 'num_passes': 3}}
        visits = layer_visits(config)
        self.assertEqual(visits, [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(visits), 16)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
