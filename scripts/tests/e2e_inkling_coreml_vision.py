#!/usr/bin/env python3
"""Convert a small complete vision encoder and check varying image grids."""
import argparse
import json
from pathlib import Path
import sys

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inkling_coreml import Vision, convert_part, image_inputs
from export_inkling_coreml import compile_and_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    c = {'hidden_size': 32, 'intermediate_size': 64, 'num_attention_heads': 4,
         'num_hidden_layers': 2, 'native_patch_size': 20, 'adapter_merge_size': 2,
         'posemb_grid_size': 16, 'layer_norm_eps': 1e-5}
    rng = np.random.default_rng(97)
    weights = {}
    def random(name, shape, scale=.03):
        weights['vision_tower.' + name] = rng.normal(0, scale, shape).astype(np.float32)
    def norm(name):
        weights['vision_tower.' + name + '.weight'] = np.ones(32, dtype=np.float32)
        weights['vision_tower.' + name + '.bias'] = np.zeros(32, dtype=np.float32)
    random('embeddings.patch_embedding.weight', (32, 1200))
    random('embeddings.position_embedding', (256, 32))
    random('embeddings.class_embedding', (32,))
    random('embeddings.cls_position_embedding', (32,))
    random('proj_in.weight', (64, 128))
    random('proj_out.weight', (64, 64))
    norm('pre_layernorm')
    for index in range(2):
        prefix = f'layers.{index}.'
        for name, shape in [('self_attn.q_proj', (32, 32)), ('self_attn.k_proj', (32, 32)),
                            ('self_attn.v_proj', (32, 32)), ('self_attn.out_proj', (32, 32)),
                            ('mlp.fc1', (64, 32)), ('mlp.fc2', (32, 64))]:
            random(prefix + name + '.weight', shape)
            random(prefix + name + '.bias', (shape[0],))
        norm(prefix + 'layernorm_before'); norm(prefix + 'layernorm_after')
        for name in ['lambda_1', 'lambda_2']:
            weights['vision_tower.' + prefix + name] = np.full(32, .3, dtype=np.float32)
    module = Vision(c, weights).eval()
    package = args.output / 'vision.mlpackage'
    options = {'max_patches': 9}
    report = convert_part(module, c, 'vision', package, **options)
    report['compile_check'] = compile_and_check(package, module, c, 'vision', options,
                                               args.output / 'compiled/vision.mlmodelc')
    model = ct.models.CompiledMLModel(str(args.output / 'compiled/vision.mlmodelc'),
                                     compute_units=ct.ComputeUnit.CPU_ONLY)
    checks = []
    for height, width in [(1, 1), (40, 40), (41, 81), (80, 80), (120, 120)]:
        inputs = image_inputs(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), c, max_patches=9)
        with torch.no_grad():
            expected = module(*[torch.from_numpy(value) for value in inputs.values()]).numpy()
        actual = model.predict(inputs)['image_embeddings']
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=.02, atol=.002)
        checks.append({'image': [height, width], 'patches': len(actual),
                       'max_absolute_error': float(np.max(np.abs(actual.astype(np.float32) - expected)))})
    report['checks'] = checks
    (args.output / 'vision-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
