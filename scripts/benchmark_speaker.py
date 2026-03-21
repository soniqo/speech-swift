#!/usr/bin/env python3
"""
Speaker embedding benchmark for speech-swift.

Evaluates speaker verification (EER, minDCF) and embedding quality across
WeSpeaker (MLX/CoreML) and CAM++ (CoreML) backends.

Three evaluation modes:
1. VoxCeleb1-O speaker verification: EER, minDCF from 37,720 trial pairs
2. VoxConverse embedding quality: intra/inter speaker cosine similarity
3. Latency: extraction time per engine

Usage:
    # VoxConverse embedding quality (uses existing data)
    python scripts/benchmark_speaker.py --voxconverse

    # VoxCeleb1-O speaker verification
    python scripts/benchmark_speaker.py --voxceleb

    # Latency benchmark
    python scripts/benchmark_speaker.py --latency

    # All engines comparison
    python scripts/benchmark_speaker.py --compare

    # Download VoxCeleb1-O test data
    python scripts/benchmark_speaker.py --download-voxceleb
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VOXCONVERSE_DIR = Path("benchmarks/voxconverse")
VOXCONVERSE_AUDIO = VOXCONVERSE_DIR / "audio"
VOXCONVERSE_REF = VOXCONVERSE_DIR / "ref"

VOXCELEB_DIR = Path("benchmarks/voxceleb1")
VOXCELEB_AUDIO = VOXCELEB_DIR / "wav"
VOXCELEB_TRIALS = VOXCELEB_DIR / "veri_test2.txt"

LIBRISPEECH_DIR = Path("benchmarks/librispeech/test-clean")

RESULTS_DIR = Path("benchmarks/speaker")

VOXCELEB_TRIALS_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt"
)

ENGINES = ["mlx", "coreml", "camplusplus"]


# ---------------------------------------------------------------------------
# Embedding extraction via CLI
# ---------------------------------------------------------------------------

def extract_embedding(cli_path: str, audio_path: str, engine: str,
                      timeout: int = 120) -> tuple:
    """Extract speaker embedding via CLI. Returns (embedding, elapsed) or (None, 0)."""
    cmd = [cli_path, "embed-speaker", str(audio_path),
           "--engine", engine, "--json"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, 0

    if result.returncode != 0:
        return None, 0

    # Parse JSON output line
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{") and '"embedding"' in line:
            try:
                data = json.loads(line)
                emb = data["embedding"]
                elapsed = data.get("elapsed", 0)
                return emb, elapsed
            except (json.JSONDecodeError, KeyError):
                pass

    return None, 0


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# VoxCeleb1-O speaker verification
# ---------------------------------------------------------------------------

def download_voxceleb_trials():
    """Download the VoxCeleb1-O trial list."""
    VOXCELEB_DIR.mkdir(parents=True, exist_ok=True)

    if VOXCELEB_TRIALS.exists():
        print(f"Trial list already exists: {VOXCELEB_TRIALS}")
        return True

    print(f"Downloading VoxCeleb1-O trial list...")
    try:
        urllib.request.urlretrieve(VOXCELEB_TRIALS_URL, VOXCELEB_TRIALS)
        n_lines = sum(1 for _ in open(VOXCELEB_TRIALS))
        print(f"  Downloaded {n_lines} trial pairs")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def parse_trials(trials_path: Path) -> list:
    """Parse VoxCeleb1-O trial list: label path1 path2."""
    trials = []
    for line in open(trials_path):
        parts = line.strip().split()
        if len(parts) == 3:
            label = int(parts[0])  # 1=same, 0=different
            trials.append((label, parts[1], parts[2]))
    return trials


def compute_eer(scores: list, labels: list) -> tuple:
    """Compute Equal Error Rate and threshold.

    Returns (eer, threshold).
    """
    # Sort by score descending
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 1.0, 0.0

    # Walk through thresholds
    tp = 0
    fp = 0
    best_eer = 1.0
    best_thresh = 0.0

    prev_score = None
    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1

        if prev_score is not None and score != prev_score:
            fnr = 1.0 - tp / n_pos  # false negative rate
            fpr = fp / n_neg         # false positive rate

            if abs(fnr - fpr) < abs(best_eer - 0):
                if abs(fnr - fpr) < best_eer:
                    best_eer = (fnr + fpr) / 2
                    best_thresh = (score + prev_score) / 2

        prev_score = score

    return best_eer, best_thresh


def compute_min_dcf(scores: list, labels: list,
                    p_target: float = 0.01, c_miss: float = 1, c_fa: float = 1) -> float:
    """Compute minimum Detection Cost Function."""
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 1.0

    tp = 0
    fp = 0
    min_dcf = float("inf")

    for score, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1

        fnr = 1.0 - tp / n_pos
        fpr = fp / n_neg

        dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        min_dcf = min(min_dcf, dcf)

    # Normalize by best possible DCF
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))
    return min_dcf / c_default if c_default > 0 else min_dcf


def run_voxceleb_benchmark(cli_path: str, engine: str,
                           max_pairs: int = 0) -> dict:
    """Run VoxCeleb1-O speaker verification benchmark."""
    if not VOXCELEB_TRIALS.exists():
        if not download_voxceleb_trials():
            return {}

    if not VOXCELEB_AUDIO.exists():
        print(f"\nVoxCeleb1 test audio not found at: {VOXCELEB_AUDIO}")
        print("Download from: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html")
        print(f"Extract to: {VOXCELEB_AUDIO}/")
        print("  Expected structure: {VOXCELEB_AUDIO}/id10270/x6uYqmx31kE/00001.wav")
        return {}

    trials = parse_trials(VOXCELEB_TRIALS)
    if max_pairs > 0:
        trials = trials[:max_pairs]

    print(f"\nVoxCeleb1-O verification ({engine}): {len(trials)} pairs")

    # Collect unique utterance paths
    utterances = set()
    for _, p1, p2 in trials:
        utterances.add(p1)
        utterances.add(p2)

    print(f"  Unique utterances: {len(utterances)}")

    # Extract embeddings (with cache)
    cache_file = RESULTS_DIR / f"emb_cache_{engine}.json"
    emb_cache = {}
    if cache_file.exists():
        try:
            emb_cache = json.load(open(cache_file))
            print(f"  Loaded {len(emb_cache)} cached embeddings")
        except Exception:
            pass

    n_extracted = 0
    n_cached = 0
    n_failed = 0
    total_time = 0
    total = len(utterances)

    for idx, utt in enumerate(sorted(utterances)):
        if utt in emb_cache:
            n_cached += 1
            continue

        audio_path = VOXCELEB_AUDIO / utt
        if not audio_path.exists():
            n_failed += 1
            continue

        emb, elapsed = extract_embedding(cli_path, str(audio_path), engine)
        if emb is not None:
            emb_cache[utt] = emb
            n_extracted += 1
            total_time += elapsed
        else:
            n_failed += 1

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{total}] extracted={n_extracted} "
                  f"cached={n_cached} failed={n_failed}", flush=True)

    # Save cache
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(emb_cache, f)

    print(f"  Extraction done: {n_extracted} new, {n_cached} cached, "
          f"{n_failed} failed")
    if n_extracted > 0:
        print(f"  Mean extraction time: {total_time/n_extracted:.3f}s")

    # Score pairs
    scores = []
    labels = []
    n_skipped = 0

    for label, p1, p2 in trials:
        if p1 not in emb_cache or p2 not in emb_cache:
            n_skipped += 1
            continue
        sim = cosine_similarity(emb_cache[p1], emb_cache[p2])
        scores.append(sim)
        labels.append(label)

    if not scores:
        print("  No valid pairs to score.")
        return {}

    print(f"  Scored {len(scores)} pairs ({n_skipped} skipped)")

    eer, eer_thresh = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(scores, labels)

    # Score statistics
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    results = {
        "engine": engine,
        "dataset": "voxceleb1-o",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_pairs": len(scores),
        "num_positive": len(pos_scores),
        "num_negative": len(neg_scores),
        "eer": round(eer * 100, 2),
        "eer_threshold": round(eer_thresh, 4),
        "min_dcf": round(min_dcf, 4),
        "positive_mean_sim": round(sum(pos_scores) / len(pos_scores), 4) if pos_scores else 0,
        "negative_mean_sim": round(sum(neg_scores) / len(neg_scores), 4) if neg_scores else 0,
        "mean_extraction_time": round(total_time / max(n_extracted, 1), 3),
    }

    return results


# ---------------------------------------------------------------------------
# LibriSpeech speaker verification
# ---------------------------------------------------------------------------

def generate_librispeech_trials(audio_dir: Path,
                                max_pos: int = 2000,
                                max_neg: int = 2000) -> list:
    """Generate trial pairs from LibriSpeech test-clean directory structure.

    Directory layout: speaker_id/chapter_id/speaker-chapter-utterance.flac
    Returns list of (label, path1, path2) tuples.
    """
    import random
    random.seed(42)

    # Collect utterances per speaker
    speaker_utts = defaultdict(list)
    for flac in sorted(audio_dir.rglob("*.flac")):
        speaker_id = flac.parts[-3]  # speaker_id directory
        speaker_utts[speaker_id].append(str(flac))

    speakers = sorted(speaker_utts.keys())
    print(f"  {len(speakers)} speakers, "
          f"{sum(len(v) for v in speaker_utts.values())} utterances")

    # Positive pairs: same speaker, different utterances
    pos_pairs = []
    for spk in speakers:
        utts = speaker_utts[spk]
        if len(utts) < 2:
            continue
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                pos_pairs.append((1, utts[i], utts[j]))
    random.shuffle(pos_pairs)
    pos_pairs = pos_pairs[:max_pos]

    # Negative pairs: different speakers
    neg_pairs = []
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            u1 = random.choice(speaker_utts[speakers[i]])
            u2 = random.choice(speaker_utts[speakers[j]])
            neg_pairs.append((0, u1, u2))
    random.shuffle(neg_pairs)
    neg_pairs = neg_pairs[:max_neg]

    trials = pos_pairs + neg_pairs
    random.shuffle(trials)
    return trials


def run_librispeech_benchmark(cli_path: str, engine: str,
                              max_pairs: int = 4000) -> dict:
    """Run speaker verification on LibriSpeech test-clean."""
    if not LIBRISPEECH_DIR.exists():
        print("LibriSpeech test-clean not found. Run benchmark_asr.py first.")
        return {}

    half = max_pairs // 2
    print(f"\nLibriSpeech verification ({engine}): generating trials...")
    trials = generate_librispeech_trials(
        LIBRISPEECH_DIR, max_pos=half, max_neg=half)

    if not trials:
        print("  No trials generated.")
        return {}

    print(f"  {len(trials)} trial pairs "
          f"({sum(1 for t in trials if t[0]==1)} pos, "
          f"{sum(1 for t in trials if t[0]==0)} neg)")

    # Collect unique utterance paths
    utterances = set()
    for _, p1, p2 in trials:
        utterances.add(p1)
        utterances.add(p2)

    print(f"  {len(utterances)} unique utterances to embed")

    # Extract embeddings (with cache)
    cache_file = RESULTS_DIR / f"emb_cache_libri_{engine}.json"
    emb_cache = {}
    if cache_file.exists():
        try:
            emb_cache = json.load(open(cache_file))
            print(f"  Loaded {len(emb_cache)} cached embeddings")
        except Exception:
            pass

    n_extracted = 0
    n_cached = 0
    n_failed = 0
    total_time = 0
    total = len(utterances)

    for idx, utt in enumerate(sorted(utterances)):
        if utt in emb_cache:
            n_cached += 1
            continue

        emb, elapsed = extract_embedding(cli_path, utt, engine)
        if emb is not None:
            emb_cache[utt] = emb
            n_extracted += 1
            total_time += elapsed
        else:
            n_failed += 1

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{total}] extracted={n_extracted} "
                  f"cached={n_cached} failed={n_failed}", flush=True)

    # Save cache
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(emb_cache, f)

    print(f"  Done: {n_extracted} new, {n_cached} cached, {n_failed} failed")

    # Score pairs
    scores = []
    labels = []
    for label, p1, p2 in trials:
        if p1 not in emb_cache or p2 not in emb_cache:
            continue
        sim = cosine_similarity(emb_cache[p1], emb_cache[p2])
        scores.append(sim)
        labels.append(label)

    if not scores:
        return {}

    eer, eer_thresh = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(scores, labels)

    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    results = {
        "engine": engine,
        "dataset": "librispeech-test-clean",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_pairs": len(scores),
        "num_positive": len(pos_scores),
        "num_negative": len(neg_scores),
        "num_speakers": len(set(
            Path(p).parts[-3] for _, p, _ in trials)),
        "eer": round(eer * 100, 2),
        "eer_threshold": round(eer_thresh, 4),
        "min_dcf": round(min_dcf, 4),
        "positive_mean_sim": round(
            sum(pos_scores) / len(pos_scores), 4) if pos_scores else 0,
        "negative_mean_sim": round(
            sum(neg_scores) / len(neg_scores), 4) if neg_scores else 0,
        "mean_extraction_time": round(
            total_time / max(n_extracted, 1), 3),
    }

    return results


# ---------------------------------------------------------------------------
# VoxConverse embedding quality
# ---------------------------------------------------------------------------

def extract_speaker_segments(rttm_path: Path, audio_path: Path,
                             min_duration: float = 1.0) -> dict:
    """Parse RTTM and return {speaker: [(start, end), ...]} for segments >= min_duration."""
    speakers = defaultdict(list)
    for line in open(rttm_path):
        parts = line.strip().split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        speaker = parts[7]
        start = float(parts[3])
        dur = float(parts[4])
        if dur >= min_duration:
            speakers[speaker].append((start, start + dur))
    return dict(speakers)


def extract_segment_embedding(cli_path: str, audio_path: str,
                              start: float, end: float,
                              engine: str) -> tuple:
    """Extract embedding from an audio segment using ffmpeg + CLI."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Extract segment using ffmpeg
        duration = end - start
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ss", str(start), "-t", str(duration),
             "-ar", "16000", "-ac", "1", tmp_path],
            capture_output=True, timeout=30, check=True
        )
        emb, elapsed = extract_embedding(cli_path, tmp_path, engine)
        return emb, elapsed
    except Exception:
        return None, 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_voxconverse_benchmark(cli_path: str, engine: str,
                              num_files: int = 0,
                              max_segments_per_speaker: int = 5) -> dict:
    """Evaluate embedding discriminability on VoxConverse."""
    if not VOXCONVERSE_AUDIO.exists() or not VOXCONVERSE_REF.exists():
        print("VoxConverse data not found. Run benchmark_diarization.py "
              "--download-only first.")
        return {}

    ref_files = sorted(VOXCONVERSE_REF.glob("*.rttm"))
    if num_files > 0:
        ref_files = ref_files[:num_files]

    print(f"\nVoxConverse embedding quality ({engine}): {len(ref_files)} files")

    all_intra = []  # cosine sims between segments of same speaker
    all_inter = []  # cosine sims between segments of different speakers
    total_elapsed = 0
    total_extractions = 0

    for ref_path in ref_files:
        name = ref_path.stem
        audio_path = VOXCONVERSE_AUDIO / f"{name}.wav"
        if not audio_path.exists():
            print(f"  Skipping {name} (no audio)")
            continue

        print(f"  {name}...", end=" ", flush=True)

        # Get speaker segments from reference RTTM
        speaker_segments = extract_speaker_segments(ref_path, audio_path)

        if len(speaker_segments) < 2:
            print(f"only {len(speaker_segments)} speaker(s), skipping")
            continue

        # Extract embeddings for each speaker's segments
        speaker_embeddings = {}
        for spk, segments in speaker_segments.items():
            # Take up to max_segments_per_speaker, prefer longer segments
            segments = sorted(segments, key=lambda s: s[1] - s[0], reverse=True)
            segments = segments[:max_segments_per_speaker]

            embs = []
            for start, end in segments:
                emb, elapsed = extract_segment_embedding(
                    cli_path, str(audio_path), start, end, engine)
                if emb is not None:
                    embs.append(emb)
                    total_elapsed += elapsed
                    total_extractions += 1

            if embs:
                speaker_embeddings[spk] = embs

        if len(speaker_embeddings) < 2:
            print(f"only {len(speaker_embeddings)} speaker(s) with embeddings")
            continue

        # Compute intra-speaker similarities
        for spk, embs in speaker_embeddings.items():
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sim = cosine_similarity(embs[i], embs[j])
                    all_intra.append(sim)

        # Compute inter-speaker similarities
        speakers = list(speaker_embeddings.keys())
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                embs_i = speaker_embeddings[speakers[i]]
                embs_j = speaker_embeddings[speakers[j]]
                # Compare first embedding of each speaker
                for ei in embs_i[:2]:
                    for ej in embs_j[:2]:
                        sim = cosine_similarity(ei, ej)
                        all_inter.append(sim)

        n_spk = len(speaker_embeddings)
        n_emb = sum(len(e) for e in speaker_embeddings.values())
        print(f"{n_spk} speakers, {n_emb} embeddings")

    if not all_intra or not all_inter:
        print("  Not enough data for analysis.")
        return {}

    def stats(values):
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "count": n,
        }

    intra_stats = stats(all_intra)
    inter_stats = stats(all_inter)
    separation = intra_stats["mean"] - inter_stats["mean"]

    results = {
        "engine": engine,
        "dataset": "voxconverse-test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_files": len(ref_files),
        "intra_speaker": intra_stats,
        "inter_speaker": inter_stats,
        "separation": round(separation, 4),
        "total_extractions": total_extractions,
        "mean_extraction_time": round(total_elapsed / max(total_extractions, 1), 3),
    }

    return results


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def run_latency_benchmark(cli_path: str, engine: str,
                          audio_path: str = None,
                          iterations: int = 10) -> dict:
    """Measure embedding extraction latency."""
    if audio_path is None:
        # Use test audio
        test_audio = Path("Tests/Qwen3ASRTests/Resources/test_audio.wav")
        if not test_audio.exists():
            print("Test audio not found.")
            return {}
        audio_path = str(test_audio)

    print(f"\nLatency benchmark ({engine}): {iterations} iterations")

    # Warmup
    print("  Warmup...", end=" ", flush=True)
    emb, _ = extract_embedding(cli_path, audio_path, engine)
    if emb is None:
        print("FAILED")
        return {}
    print(f"OK (dim={len(emb)})")

    # Timed iterations
    times = []
    for i in range(iterations):
        emb, elapsed = extract_embedding(cli_path, audio_path, engine)
        if emb is not None:
            times.append(elapsed)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{iterations}] {elapsed:.3f}s")

    if not times:
        return {}

    mean_t = sum(times) / len(times)
    std_t = math.sqrt(sum((t - mean_t) ** 2 for t in times) / len(times))

    results = {
        "engine": engine,
        "embedding_dim": len(emb) if emb else 0,
        "iterations": len(times),
        "mean_ms": round(mean_t * 1000, 1),
        "std_ms": round(std_t * 1000, 1),
        "min_ms": round(min(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
    }

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_verification_summary(results: dict):
    """Print speaker verification results."""
    print(f"\n{'='*60}")
    print(f"Speaker Verification: {results['dataset']}")
    print(f"Engine: {results['engine']}")
    print(f"{'='*60}")
    print(f"  Trial pairs:        {results['num_pairs']}")
    print(f"  Positive pairs:     {results['num_positive']}")
    print(f"  Negative pairs:     {results['num_negative']}")
    print(f"  EER:                {results['eer']:.2f}%")
    print(f"  EER threshold:      {results['eer_threshold']:.4f}")
    print(f"  minDCF (p=0.01):    {results['min_dcf']:.4f}")
    print(f"  Positive mean sim:  {results['positive_mean_sim']:.4f}")
    print(f"  Negative mean sim:  {results['negative_mean_sim']:.4f}")
    print(f"{'='*60}")


def print_voxconverse_summary(results: dict):
    """Print VoxConverse embedding quality results."""
    print(f"\n{'='*60}")
    print(f"Embedding Quality: VoxConverse")
    print(f"Engine: {results['engine']}")
    print(f"{'='*60}")
    intra = results["intra_speaker"]
    inter = results["inter_speaker"]
    print(f"  Intra-speaker similarity:")
    print(f"    Mean: {intra['mean']:.4f} ± {intra['std']:.4f} "
          f"[{intra['min']:.4f}, {intra['max']:.4f}] (n={intra['count']})")
    print(f"  Inter-speaker similarity:")
    print(f"    Mean: {inter['mean']:.4f} ± {inter['std']:.4f} "
          f"[{inter['min']:.4f}, {inter['max']:.4f}] (n={inter['count']})")
    print(f"  Separation (intra - inter): {results['separation']:.4f}")
    print(f"  Extractions: {results['total_extractions']} "
          f"({results['mean_extraction_time']:.3f}s mean)")
    print(f"{'='*60}")


def print_latency_summary(results: dict):
    """Print latency benchmark results."""
    print(f"\n{'='*60}")
    print(f"Latency: {results['engine']} ({results['embedding_dim']}-dim)")
    print(f"{'='*60}")
    print(f"  Mean:  {results['mean_ms']:.1f} ms ± {results['std_ms']:.1f}")
    print(f"  Min:   {results['min_ms']:.1f} ms")
    print(f"  Max:   {results['max_ms']:.1f} ms")
    print(f"{'='*60}")


def print_comparison_table(all_latency: list, all_voxconverse: list):
    """Print comparison table across engines."""
    if all_latency:
        print(f"\n{'='*70}")
        print(f"Latency Comparison")
        print(f"{'='*70}")
        print(f"{'Engine':<16} {'Dim':>5} {'Mean (ms)':>10} {'Std (ms)':>10} "
              f"{'Min (ms)':>10}")
        print(f"{'-'*16} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
        for r in all_latency:
            print(f"{r['engine']:<16} {r['embedding_dim']:>5} "
                  f"{r['mean_ms']:>10.1f} {r['std_ms']:>10.1f} "
                  f"{r['min_ms']:>10.1f}")

    if all_voxconverse:
        print(f"\n{'='*70}")
        print(f"Embedding Quality Comparison (VoxConverse)")
        print(f"{'='*70}")
        print(f"{'Engine':<16} {'Intra':>8} {'Inter':>8} {'Sep':>8} "
              f"{'Extractions':>12}")
        print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
        for r in all_voxconverse:
            print(f"{r['engine']:<16} "
                  f"{r['intra_speaker']['mean']:>8.4f} "
                  f"{r['inter_speaker']['mean']:>8.4f} "
                  f"{r['separation']:>8.4f} "
                  f"{r['total_extractions']:>12}")

    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Speaker embedding benchmark (WeSpeaker + CAM++)")
    parser.add_argument("--cli-path", default=".build/release/audio")
    parser.add_argument("--engine", default="mlx",
                        choices=ENGINES,
                        help="Engine: mlx, coreml (WeSpeaker), camplusplus (CAM++)")

    # Evaluation modes
    parser.add_argument("--librispeech", action="store_true",
                        help="Run LibriSpeech test-clean verification (EER, minDCF)")
    parser.add_argument("--voxceleb", action="store_true",
                        help="Run VoxCeleb1-O speaker verification (EER, minDCF)")
    parser.add_argument("--voxconverse", action="store_true",
                        help="Run VoxConverse embedding quality analysis")
    parser.add_argument("--latency", action="store_true",
                        help="Run latency benchmark")
    parser.add_argument("--compare", action="store_true",
                        help="Run all engines and compare")

    # Options
    parser.add_argument("--num-files", type=int, default=0,
                        help="Limit VoxConverse files (0 = all)")
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Limit VoxCeleb trial pairs (0 = all)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Latency benchmark iterations")
    parser.add_argument("--download-voxceleb", action="store_true",
                        help="Download VoxCeleb1-O trial list")

    args = parser.parse_args()

    if args.download_voxceleb:
        download_voxceleb_trials()
        print("\nTo complete VoxCeleb1-O benchmark, download test audio from:")
        print("  https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html")
        print(f"  Extract to: {VOXCELEB_AUDIO}/")
        return

    if not Path(args.cli_path).exists():
        print(f"CLI not found: {args.cli_path}. Build with: make build")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Default: run all available benchmarks
    if not (args.librispeech or args.voxceleb or args.voxconverse
            or args.latency or args.compare):
        args.librispeech = True
        args.voxconverse = True
        args.latency = True

    if args.compare:
        all_latency = []
        all_voxconverse = []
        all_verification = []

        for engine in ENGINES:
            print(f"\n{'#'*60}")
            print(f"# Engine: {engine}")
            print(f"{'#'*60}")

            # Latency
            lat = run_latency_benchmark(
                args.cli_path, engine, iterations=args.iterations)
            if lat:
                all_latency.append(lat)
                print_latency_summary(lat)

            # VoxConverse
            vc = run_voxconverse_benchmark(
                args.cli_path, engine, args.num_files)
            if vc:
                all_voxconverse.append(vc)
                print_voxconverse_summary(vc)

            # LibriSpeech verification
            if LIBRISPEECH_DIR.exists():
                ls = run_librispeech_benchmark(
                    args.cli_path, engine, args.max_pairs or 4000)
                if ls:
                    all_verification.append(ls)
                    print_verification_summary(ls)

            # VoxCeleb (if audio available)
            if VOXCELEB_AUDIO.exists():
                vx = run_voxceleb_benchmark(
                    args.cli_path, engine, args.max_pairs)
                if vx:
                    all_verification.append(vx)
                    print_verification_summary(vx)

        print_comparison_table(all_latency, all_voxconverse)

        # Save all results
        all_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency": all_latency,
            "voxconverse": all_voxconverse,
            "verification": all_verification,
        }
        out_file = RESULTS_DIR / "results_comparison.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_file}")
        return

    # Single engine runs
    if args.latency:
        lat = run_latency_benchmark(
            args.cli_path, args.engine, iterations=args.iterations)
        if lat:
            print_latency_summary(lat)
            out_file = RESULTS_DIR / f"results_latency_{args.engine}.json"
            with open(out_file, "w") as f:
                json.dump(lat, f, indent=2)

    if args.voxconverse:
        vc = run_voxconverse_benchmark(
            args.cli_path, args.engine, args.num_files)
        if vc:
            print_voxconverse_summary(vc)
            out_file = RESULTS_DIR / f"results_voxconverse_{args.engine}.json"
            with open(out_file, "w") as f:
                json.dump(vc, f, indent=2)

    if args.librispeech:
        ls = run_librispeech_benchmark(
            args.cli_path, args.engine, args.max_pairs or 4000)
        if ls:
            print_verification_summary(ls)
            out_file = RESULTS_DIR / f"results_librispeech_{args.engine}.json"
            with open(out_file, "w") as f:
                json.dump(ls, f, indent=2)

    if args.voxceleb:
        vx = run_voxceleb_benchmark(
            args.cli_path, args.engine, args.max_pairs)
        if vx:
            print_verification_summary(vx)
            out_file = RESULTS_DIR / f"results_voxceleb_{args.engine}.json"
            with open(out_file, "w") as f:
                json.dump(vx, f, indent=2)


if __name__ == "__main__":
    main()
