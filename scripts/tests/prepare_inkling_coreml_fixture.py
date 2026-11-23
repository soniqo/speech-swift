#!/usr/bin/env python3
"""Dequantize the checked-in tiny MLX fixture on the CPU for Core ML tests."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / 'Tests/Qwen3ChatTests/Fixtures/Inkling2'
    config = json.loads((source / 'config.json').read_text())
    mx.set_default_device(mx.cpu)
    tensors = mx.load(str(source / 'model.safetensors'))
    quantization = config['quantization']
    result = {}
    for name, value in tensors.items():
        if name.endswith(('.scales', '.biases')):
            continue
        prefix = name.removesuffix('.weight')
        if name.endswith('.weight') and prefix + '.scales' in tensors:
            settings = quantization.get(prefix, quantization)
            value = mx.dequantize(value, tensors[prefix + '.scales'], tensors[prefix + '.biases'],
                                 group_size=settings['group_size'], bits=settings['bits'])
        result[name] = np.array(value.astype(mx.float32))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'weights.npz', **result)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(f'Prepared {len(result)} CPU tensors from the checked-in fixture.')


if __name__ == '__main__':
    main()
