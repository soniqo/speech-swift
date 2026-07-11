#!/usr/bin/env python3
"""Probe MAGNeT LM first-forward logits at stage=0 on an all-mask sequence.

Mirrors the assertions in
  Tests/MAGNeTMusicGenTests/E2EDiagnosticsTests.swift::testLMFirstForwardMatchesPython

Run from the magnet/export directory inside the poetry env:

  poetry run python /tmp/probe_magnet_logits.py --bundle <path-to-bundle>

Prints the eight first cond/uncond logits and the stage-0 argmax histogram so
you can paste the numbers into the Swift parity test for a different variant.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import mlx.core as mx

from test_mlx import load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to bundle dir (config.json + model.safetensors)")
    ap.add_argument("--text", default="happy rock")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    print(f"Loading {bundle} ...")
    model, cfg = load(bundle)

    # Step 1: text -> conditioning, build CFG batch.
    cond = model.text_conditioner(args.text)        # (1, L, D)
    uncond = mx.zeros_like(cond)
    conditioning = mx.concatenate([cond, uncond], axis=0)  # (2, L, D)

    # Step 2: all-mask audio token grid at the model's seq_len.
    T = model.lm._seq_len
    K = model.num_codebooks
    mask_id = model.lm.mask_token_id
    gen = mx.full((1, K, T), mask_id, dtype=mx.int32)

    # Step 3: LM forward at stage=0 on (lm_input transposed to (B, T, K)).
    lm_input = mx.concatenate([gen, gen], axis=0).transpose(0, 2, 1)  # (2, T, K)
    all_logits = model.lm(lm_input, conditioning, stage=0)             # (2, T, K, card)
    mx.eval(all_logits)

    cond_logits = all_logits[:1]
    uncond_logits = all_logits[1:2]

    print(f"\n[PY] LM cond_logits[0,0,0,:8]   = {cond_logits[0, 0, 0, :8].tolist()}")
    print(f"[PY] LM uncond_logits[1,0,0,:8] = {uncond_logits[0, 0, 0, :8].tolist()}")

    # Argmax histogram across T for codebook 0 of cond.
    argmax = mx.argmax(cond_logits[0, :, 0, :], axis=-1).tolist()
    hist = Counter(argmax).most_common(5)
    print("\n[PY] stage-0 argmax histogram (top 5):")
    for tok, count in hist:
        print(f"    token {tok}: {count} times ({100 * count / T:.1f}%)")

    # Single-line summary for copy-paste into the Swift test docstring.
    flat_cond = cond_logits[0, 0, 0, :8].tolist()
    flat_uncond = uncond_logits[0, 0, 0, :8].tolist()
    print(f"\n[PY-SUMMARY] cond_logits[0,0,0,:8] = {[round(v, 4) for v in flat_cond]}")
    print(f"[PY-SUMMARY] uncond_logits[1,0,0,:8] = {[round(v, 4) for v in flat_uncond]}")


if __name__ == "__main__":
    main()
