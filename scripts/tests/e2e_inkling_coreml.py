#!/usr/bin/env python3
"""Check all decoder visits through real Core ML partitions on the tiny fixture.

Prepare weights.npz by dequantizing the checked-in MLX fixture on the CPU. No
production checkpoint or download is needed for this structural parity test.
"""
from pathlib import Path
import argparse
import copy
import gc
import json
import sys

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling_coreml import Embedding, DecoderLayer, OutputHead, Decoder, convert_part, conversion_inputs
from export_inkling_coreml import FORMAT, compile_and_check


class CoreMLPart:
    def __init__(self, path, inputs, outputs, package):
        self.model = ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
        self.inputs, self.outputs = inputs, outputs
        self.name = path.stem
        self.required = {value.name for value in ct.utils.load_spec(str(package)).description.input}

    def __call__(self, *args):
        values = {name: np.ascontiguousarray(value.numpy()) for name, value in zip(self.inputs, args)}
        try:
            result = self.model.predict({name: value for name, value in values.items() if name in self.required})
        except Exception as error:
            raise RuntimeError(f'{self.name}: {[(name, value.shape) for name, value in values.items()]}') from error
        converted = tuple(torch.from_numpy(result[name]) for name in self.outputs)
        return converted[0] if len(converted) == 1 else converted


def verify_decoder_equations(config, weights, reference):
    """Check the port's equations against the pinned independent MLX goldens."""
    options = {'quantized': False, 'dtype': torch.float32}
    decoder = Decoder(config, Embedding(config, weights, **options),
        [DecoderLayer(config, layer, weights, **options) for layer in range(config['num_hidden_layers'])],
        OutputHead(config, weights, **options), dtype=torch.float32)
    results = []

    def compare(name, actual):
        expected = np.asarray(reference[name], dtype=np.float32).reshape(actual.shape)
        actual = actual.numpy()
        error = actual - expected
        nrmse = float(np.sqrt(np.mean(error ** 2)) / np.sqrt(np.mean(expected ** 2)))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
        assert nrmse < 1e-5
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        results.append({'name': name, 'max_absolute_error': float(np.abs(error).max()),
                        'normalized_rmse': nrmse, 'greedy_choices_match': True})

    with torch.no_grad():
        compare('full_logits', decoder.forward(torch.tensor([reference['tokens']], dtype=torch.int32)))
        compare('continued_logits', decoder.forward(torch.tensor([[8]], dtype=torch.int32)))
        decoder.reset()
        for token in reference['tokens']:
            last = decoder.forward(torch.tensor([[token]], dtype=torch.int32))
        compare('sequential_logits', last)
        decoder.reset()
        compare('grouped_logits', decoder.forward(torch.tensor([reference['grouped_tokens']], dtype=torch.int32)))
    return {'reference_revision': reference['reference_revision'], 'checks': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', action='store_true', help='Reuse already converted local tiny partitions.')
    parser.add_argument('--activation-dtype', choices=['float16', 'float32'], default='float32')
    args = parser.parse_args()
    torch.set_num_threads(2)
    dtype = torch.float32 if args.activation_dtype == 'float32' else torch.float16
    root = json.loads((args.fixture_dir / 'config.json').read_text())
    c = root['text_config']
    weights = dict(np.load(args.fixture_dir / 'weights.npz'))
    args.output.mkdir(parents=True, exist_ok=True)
    reference_path = Path(__file__).resolve().parents[2] / 'Tests/Qwen3ChatTests/Fixtures/Inkling2/reference.json'
    reference = json.loads(reference_path.read_text())
    equations = verify_decoder_equations(c, weights, reference)
    (args.output / 'independent-math-validation.json').write_text(json.dumps(equations, indent=2) + '\n')
    modules, adapters, parts = {}, {}, {}
    plan = [('embedding', 'embedding', None)] + [(f'layer-{i}', 'layer', i) for i in range(c['num_hidden_layers'])]
    plan += [('head', 'head', None)]
    for name, kind, layer in plan:
        options = {'quantized': True, 'dtype': dtype, 'group_size': 32}
        module = (Embedding(c, weights, **options) if kind == 'embedding'
                  else OutputHead(c, weights, **options) if kind == 'head'
                  else DecoderLayer(c, layer, weights, **options))
        package = args.output / f'{name}.mlpackage'
        conversion = {'context_limit': 64, 'prefill_size': 1, 'dtype': dtype}
        if layer is not None:
            conversion['layer'] = layer
        print(f'Converting and checking {name}', flush=True)
        compiled = args.output / 'compiled' / f'{name}.mlmodelc'
        if args.reuse and compiled.is_dir() and package.is_dir():
            _, sample, names = conversion_inputs(c, kind, **conversion)
            report = {'inputs': list(sample), 'outputs': names}
        else:
            report = convert_part(module, c, kind, package, **conversion)
            report['compile_check'] = compile_and_check(package, module, c, kind, conversion, compiled)
        report['package'] = package.name
        report['compiled'] = str(compiled.relative_to(args.output))
        parts[name] = report
        modules[name] = module
        adapters[name] = CoreMLPart(compiled, report['inputs'], report['outputs'], package)
        gc.collect()
    def decoder(source):
        return Decoder(c, source['embedding'], [source[f'layer-{i}'] for i in range(c['num_hidden_layers'])],
                       source['head'], dtype=dtype)
    eager, native = decoder(modules), decoder(adapters)
    checks, vectors = [], []
    def compare(tokens, name):
        expected_steps, actual_steps = [], []
        with torch.no_grad():
            for token in tokens:
                ids = torch.tensor([[token]], dtype=torch.int32)
                expected_steps.append(eager.forward(ids))
                actual_steps.append(native.forward(ids))
        expected = torch.cat(expected_steps, dim=1)
        actual = torch.cat(actual_steps, dim=1)
        errors = (actual.float() - expected.float()).abs()
        normalized_rmse = float(torch.mean(errors.square()).sqrt() / torch.mean(expected.float().square()).sqrt())
        # FP16 matrix kernels accumulate differently from PyTorch; constrain
        # both the aggregate error and individual logits, plus greedy choices.
        fp32 = dtype == torch.float32
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4 if fp32 else .02,
                                   atol=2e-5 if fp32 else .01)
        assert normalized_rmse < (1e-4 if fp32 else .01), normalized_rmse
        np.testing.assert_array_equal(actual.argmax(-1).numpy(), expected.argmax(-1).numpy())
        worst_state_rmse, worst_state_error = 0., 0.
        for actual_state, expected_state in zip(native.states, eager.states):
            for slot, (a, b) in enumerate(zip(actual_state, expected_state)):
                if slot >= 2:
                    assert a.dtype == torch.float32
                assert a.shape == b.shape
                error = (a.float() - b.float()).abs()
                relative = float(error.square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-6))
                worst_state_error = max(worst_state_error, float(error.max()))
                worst_state_rmse = max(worst_state_rmse, relative)
                assert float(error.max()) < (5e-4 if fp32 else .05), (name, slot, float(error.max()))
                assert relative < (1e-3 if fp32 else .02), (name, slot, relative)
        checks.append({'name': name, 'tokens': len(tokens), 'max_absolute_error': float(errors.max()),
                       'normalized_rmse': normalized_rmse, 'greedy_choices_match': True,
                       'state_max_absolute_error': worst_state_error, 'state_normalized_rmse': worst_state_rmse,
                       'logical_states': len(native.states), 'history_dtype': 'float32'})
        vectors.append({'name': name, 'tokens': tokens, 'torch_logits': expected.float().tolist(),
                        'coreml_logits': actual.float().tolist()})
        print(json.dumps(checks[-1]), flush=True)
    compare(reference['tokens'], 'prefill')
    compare([8], 'cached_decode')
    eager.reset(); native.reset()
    for index, token in enumerate(reference['tokens']):
        compare([token], f'sequential_{index}')
    eager.reset(); native.reset()
    compare(reference['grouped_tokens'], 'grouped_prefill')
    eager.reset(); native.reset()
    for start, end in [(0, 13), (13, 30), (30, 40)]:
        compare(reference['grouped_tokens'][start:end], f'chunk_{start}_{end}')
    architecture = copy.deepcopy(root)
    architecture['coreml_format'] = FORMAT
    architecture['quantization'] = {'bits': 8, 'group_size': 32, 'mode': 'affine'}
    (args.output / 'config.json').write_text(json.dumps(architecture, indent=2) + '\n')
    manifest = {'format': FORMAT, 'status': 'converted', 'validation_status': 'tiny_partition_and_pipeline_passed',
        'weight_bits': 8, 'activation_dtype': args.activation_dtype, 'convolution_history_dtype': 'float32',
        'context_limit': 64, 'prefill_size': 1, 'image_patch_limit': 0,
        'physical_layers': c['num_hidden_layers'], 'parts': parts, 'checks': checks}
    (args.output / 'coreml-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'tiny-validation.json').write_text(json.dumps({'checks': checks, 'vectors': vectors}, indent=2) + '\n')
    print(f'Passed {len(checks)} full/cache/chunk comparisons with separate visit histories.')


if __name__ == '__main__':
    main()
