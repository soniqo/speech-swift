#!/usr/bin/env python3
"""Grid-search FireRedVAD threshold and smoothing parameters on VoxConverse.

Tests combinations of:
  - speechThreshold: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  - smoothWindowSize: [1, 3, 5, 7, 9, 11]
  - energy-filter: [off, on]

Reports F1, FAR, MR for each combination, sorted by F1.

Usage:
    python scripts/tune_firered_vad.py [--num-files 5] [--cli-path .build/release/audio]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_DIR = Path("benchmarks/voxconverse")
AUDIO_DIR = BENCHMARK_DIR / "audio"
REF_DIR = BENCHMARK_DIR / "ref"


def parse_rttm(content):
    segments = []
    for line in content.strip().split("\n"):
        parts = line.split()
        if len(parts) < 5 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        segments.append((start, start + duration))
    return segments


def segments_to_frames(segments, duration, frame_shift=0.01):
    n_frames = int(duration / frame_shift) + 1
    frames = [0] * n_frames
    for start, end in segments:
        for i in range(max(0, int(start / frame_shift)), min(n_frames, int(end / frame_shift))):
            frames[i] = 1
    return frames


def compute_metrics(ref_segments, hyp_segments, duration):
    ref = segments_to_frames(ref_segments, duration)
    hyp = segments_to_frames(hyp_segments, duration)
    n = min(len(ref), len(hyp))
    tp = sum(1 for i in range(n) if ref[i] == 1 and hyp[i] == 1)
    fp = sum(1 for i in range(n) if ref[i] == 0 and hyp[i] == 1)
    fn = sum(1 for i in range(n) if ref[i] == 1 and hyp[i] == 0)
    tn = sum(1 for i in range(n) if ref[i] == 0 and hyp[i] == 0)
    return tp, fp, fn, tn


def run_vad(cli_path, audio_path, threshold, smooth, energy_filter, timeout=300):
    cmd = [cli_path, "vad", str(audio_path), "--engine", "firered",
           "--threshold", str(threshold), "--smooth-window", str(smooth)]
    if energy_filter:
        cmd.append("--energy-filter")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return []

    segments = []
    for line in result.stdout.split("\n"):
        m = re.search(r"\[([\d.]+)s\s*[-–]\s*([\d.]+)s\]", line)
        if m:
            segments.append((float(m.group(1)), float(m.group(2))))
    return segments


def get_duration(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return (os.path.getsize(str(path)) - 44) / (16000 * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-path", default=".build/release/audio")
    parser.add_argument("--num-files", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    ref_files = sorted(REF_DIR.glob("*.rttm"))[:args.num_files]
    if not ref_files:
        print("No RTTM files found. Run benchmark_diarization.py --download-only first.")
        sys.exit(1)

    # Load reference data
    files = []
    for ref_path in ref_files:
        name = ref_path.stem
        audio_path = AUDIO_DIR / f"{name}.wav"
        if not audio_path.exists():
            continue
        ref_segments = parse_rttm(ref_path.read_text())
        duration = get_duration(str(audio_path))
        files.append((name, str(audio_path), ref_segments, duration))

    print(f"Testing {len(files)} files")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    smooth_windows = [1, 3, 5, 7, 9, 11]
    energy_filters = [False, True]

    results = []
    total_combos = len(thresholds) * len(smooth_windows) * len(energy_filters)
    combo_idx = 0

    for ef in energy_filters:
        for thresh in thresholds:
            for smooth in smooth_windows:
                combo_idx += 1
                ef_label = "+ef" if ef else ""
                label = f"t={thresh} s={smooth}{ef_label}"
                print(f"  [{combo_idx}/{total_combos}] {label}...", end=" ", flush=True)

                total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
                ok = True

                for name, audio_path, ref_segments, duration in files:
                    try:
                        hyp = run_vad(args.cli_path, audio_path, thresh, smooth, ef, args.timeout)
                    except subprocess.TimeoutExpired:
                        ok = False
                        break
                    tp, fp, fn, tn = compute_metrics(ref_segments, hyp, duration)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_tn += tn

                if not ok:
                    print("TIMEOUT")
                    continue

                prec = total_tp / max(total_tp + total_fp, 1)
                rec = total_tp / max(total_tp + total_fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8) * 100
                far = total_fp / max(total_fp + total_tn, 1) * 100
                mr = total_fn / max(total_tp + total_fn, 1) * 100

                results.append({
                    "threshold": thresh,
                    "smooth": smooth,
                    "energy_filter": ef,
                    "f1": round(f1, 2),
                    "far": round(far, 2),
                    "mr": round(mr, 2),
                    "precision": round(prec * 100, 2),
                    "recall": round(rec * 100, 2),
                })
                print(f"F1={f1:.1f}% FAR={far:.1f}% MR={mr:.1f}%")

    # Sort by F1 descending
    results.sort(key=lambda r: r["f1"], reverse=True)

    print(f"\n{'='*80}")
    print(f"Top 10 configurations (sorted by F1):")
    print(f"{'='*80}")
    print(f"{'Threshold':>10} {'Smooth':>7} {'EF':>4} {'F1%':>8} {'FAR%':>8} {'MR%':>8} {'Prec%':>8} {'Rec%':>8}")
    print(f"{'-'*10} {'-'*7} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results[:10]:
        ef = "yes" if r["energy_filter"] else "no"
        print(f"{r['threshold']:>10} {r['smooth']:>7} {ef:>4} "
              f"{r['f1']:>7.2f}% {r['far']:>7.2f}% {r['mr']:>7.2f}% "
              f"{r['precision']:>7.2f}% {r['recall']:>7.2f}%")

    print(f"\n{'='*80}")
    print(f"Lowest FAR configurations (top 5):")
    print(f"{'='*80}")
    far_sorted = sorted(results, key=lambda r: r["far"])
    for r in far_sorted[:5]:
        ef = "yes" if r["energy_filter"] else "no"
        print(f"t={r['threshold']} s={r['smooth']} ef={ef}: "
              f"F1={r['f1']:.1f}% FAR={r['far']:.1f}% MR={r['mr']:.1f}%")

    # Save results
    out_path = Path("benchmarks/vad/firered_tuning.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
