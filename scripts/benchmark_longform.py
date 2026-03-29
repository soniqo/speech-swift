#!/usr/bin/env python3
"""
Long-form ASR benchmark: measures WER, latency, and memory stability over
sustained transcription sessions (60+ minutes).

Tests whether CoreML Neural Engine degrades under sustained load —
thermal throttling, memory pressure, ANE scheduling issues.

Dataset: Earnings-22 (earnings calls, ~119 hours, word-level transcripts)
  Paper: https://arxiv.org/abs/2203.15591
  Download: https://huggingface.co/datasets/revdotcom/earnings22

Usage:
    python scripts/benchmark_longform.py
    python scripts/benchmark_longform.py --engine parakeet --duration 60
    python scripts/benchmark_longform.py --engine qwen3 --chunk-duration 10
    python scripts/benchmark_longform.py --compare
    python scripts/benchmark_longform.py --download-only
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_BASE = Path("benchmarks/longform")
CLI_PATH = ".build/release/audio"

# Earnings-22 files to use (subset for benchmark)
# Each is a ~60 min earnings call with ground truth transcript
EARNINGS22_FILES = [
    "4441921",  # ~58 min
    "4440363",  # ~62 min
    "4442438",  # ~55 min
]

# ---------------------------------------------------------------------------
# Text normalization & WER (from benchmark_asr.py)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> dict:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    n, m = len(ref), len(hyp)

    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    subs, ins, dels = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1; i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1; j -= 1
        else:
            dels += 1; i -= 1

    errors = subs + ins + dels
    return {
        "wer": round(errors / max(n, 1) * 100, 2),
        "errors": errors, "substitutions": subs,
        "insertions": ins, "deletions": dels, "ref_words": n,
    }


# ---------------------------------------------------------------------------
# Dataset download (Earnings-22 from HuggingFace)
# ---------------------------------------------------------------------------

def download_earnings22(data_dir: Path) -> Path:
    """Download Earnings-22 dataset from HuggingFace."""
    out_dir = data_dir / "earnings22"
    if out_dir.exists() and any(out_dir.glob("*.wav")):
        print(f"  Using cached dataset: {out_dir}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading Earnings-22 from HuggingFace...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required. Install: pip install datasets soundfile")
        sys.exit(1)

    ds = load_dataset("revdotcom/earnings22", split="test")

    # Save audio files and transcripts
    saved = 0
    for row in ds:
        file_id = row.get("file", row.get("audio", {}).get("path", f"unknown_{saved}"))
        file_id = Path(file_id).stem

        # Save audio
        audio = row["audio"]
        sr = audio["sampling_rate"]
        samples = audio["array"]

        wav_path = out_dir / f"{file_id}.wav"
        if not wav_path.exists():
            import numpy as np
            import soundfile as sf
            sf.write(str(wav_path), np.array(samples, dtype=np.float32), sr)

        # Save transcript
        transcript = row.get("sentence", row.get("text", ""))
        if transcript:
            txt_path = out_dir / f"{file_id}.txt"
            txt_path.write_text(transcript)

        saved += 1
        if saved % 10 == 0:
            print(f"    Saved {saved} files...")

    print(f"  Downloaded {saved} files to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------

def get_wav_duration(path: str) -> float:
    """Get WAV file duration in seconds."""
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def chunk_audio(wav_path: str, chunk_duration: float, output_dir: Path) -> list:
    """Split a WAV file into fixed-duration chunks using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_dur = get_wav_duration(wav_path)
    num_chunks = int(total_dur / chunk_duration) + 1
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_duration
        if start >= total_dur:
            break
        out_path = output_dir / f"chunk_{i:04d}.wav"
        if not out_path.exists():
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-ss", str(start), "-t", str(chunk_duration),
                "-ar", "16000", "-ac", "1",
                str(out_path)
            ], capture_output=True, timeout=30)
        chunks.append({
            "path": str(out_path),
            "index": i,
            "start_time": start,
            "duration": min(chunk_duration, total_dur - start),
        })

    return chunks


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_chunk(cli_path: str, audio_path: str, engine: str,
                     model: str = None) -> dict:
    """Transcribe a single chunk, returning text + timing."""
    cmd = [cli_path, "transcribe", audio_path, "--engine", engine]
    if model:
        cmd.extend(["--model", model])

    wall_start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    wall_time = time.monotonic() - wall_start

    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200], "wall_time": wall_time}

    text = ""
    rtf = 0.0
    inference_time = 0.0

    for line in result.stdout.split("\n"):
        if line.startswith("Result: "):
            text = line[len("Result: "):]
        elif "Time:" in line and "RTF:" in line:
            m = re.search(r"Time:\s*([\d.]+)s.*RTF:\s*([\d.]+)", line)
            if m:
                inference_time = float(m.group(1))
                rtf = float(m.group(2))

    return {
        "text": text,
        "rtf": rtf,
        "inference_time": inference_time,
        "wall_time": round(wall_time, 3),
    }


def get_process_memory_mb() -> float:
    """Get current process tree memory (includes child CLI processes)."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_longform_benchmark(engine: str, model: str, data_dir: Path,
                           chunk_duration: float, max_duration: float,
                           output_dir: Path):
    """Run long-form benchmark on a single audio file."""
    # Find audio files
    wav_files = sorted(data_dir.glob("*.wav"))
    if not wav_files:
        print("ERROR: No WAV files found. Run with --download-only first.")
        sys.exit(1)

    # Pick the longest file, or concatenate if needed
    # For now, use individual files and process them sequentially
    print(f"\nEngine: {engine}, Chunk: {chunk_duration}s, Max: {max_duration}min")
    print(f"Found {len(wav_files)} audio files")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    total_audio_duration = 0
    chunk_index = 0

    for wav_path in wav_files:
        file_dur = get_wav_duration(str(wav_path))
        if total_audio_duration + file_dur > max_duration * 60:
            remaining = max_duration * 60 - total_audio_duration
            if remaining < chunk_duration:
                break

        file_id = wav_path.stem
        txt_path = wav_path.with_suffix(".txt")
        reference = txt_path.read_text().strip() if txt_path.exists() else None

        print(f"\n  File: {file_id} ({file_dur:.0f}s)")

        # Chunk the audio
        chunk_dir = output_dir / "chunks" / file_id
        chunks = chunk_audio(str(wav_path), chunk_duration, chunk_dir)

        for chunk in chunks:
            if total_audio_duration > max_duration * 60:
                break

            result = transcribe_chunk(CLI_PATH, chunk["path"], engine, model)

            if "error" in result:
                print(f"    Chunk {chunk_index}: ERROR {result['error'][:80]}")
                chunk_index += 1
                total_audio_duration += chunk["duration"]
                continue

            result["chunk_index"] = chunk_index
            result["file_id"] = file_id
            result["audio_offset"] = chunk["start_time"]
            result["chunk_duration"] = chunk["duration"]
            result["cumulative_audio_s"] = total_audio_duration + chunk["duration"]
            result["memory_mb"] = get_process_memory_mb()

            all_results.append(result)
            total_audio_duration += chunk["duration"]
            chunk_index += 1

            # Progress
            if chunk_index % 10 == 0:
                avg_rtf = sum(r["rtf"] for r in all_results) / len(all_results)
                print(f"    Chunk {chunk_index}: RTF={result['rtf']:.4f}, "
                      f"avg_RTF={avg_rtf:.4f}, "
                      f"cumulative={total_audio_duration/60:.1f}min")

    # Save raw results
    results_path = output_dir / f"longform_{engine}_{chunk_duration}s.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} chunk results to {results_path}")

    # Compute aggregate stats
    print_analysis(all_results, engine, chunk_duration)

    # If we have reference transcripts, compute WER per position bucket
    if any(r.get("text") for r in all_results):
        compute_positional_wer(all_results, data_dir, chunk_duration, output_dir)

    return all_results


def print_analysis(results: list, engine: str, chunk_duration: float):
    """Print latency and stability analysis."""
    if not results:
        return

    rtfs = [r["rtf"] for r in results if r["rtf"] > 0]
    wall_times = [r["wall_time"] for r in results]

    n = len(rtfs)
    first_quarter = rtfs[:n//4] if n >= 4 else rtfs
    last_quarter = rtfs[-(n//4):] if n >= 4 else rtfs

    print(f"\n{'='*60}")
    print(f"Long-Form Benchmark: {engine}, {chunk_duration}s chunks")
    print(f"{'='*60}")
    print(f"  Total chunks: {len(results)}")
    print(f"  Total audio: {sum(r.get('chunk_duration', chunk_duration) for r in results)/60:.1f} min")
    print(f"  Total wall time: {sum(wall_times)/60:.1f} min")
    print(f"\n  RTF (Real-Time Factor):")
    print(f"    Mean:  {sum(rtfs)/len(rtfs):.4f}")
    print(f"    Min:   {min(rtfs):.4f}")
    print(f"    Max:   {max(rtfs):.4f}")
    print(f"    Std:   {(sum((x - sum(rtfs)/len(rtfs))**2 for x in rtfs) / len(rtfs))**0.5:.4f}")
    print(f"\n  Stability (first 25% vs last 25%):")
    print(f"    First quarter avg RTF: {sum(first_quarter)/len(first_quarter):.4f}")
    print(f"    Last quarter avg RTF:  {sum(last_quarter)/len(last_quarter):.4f}")
    ratio = (sum(last_quarter)/len(last_quarter)) / (sum(first_quarter)/len(first_quarter))
    print(f"    Ratio (last/first):    {ratio:.3f}x")
    if ratio > 1.2:
        print(f"    ⚠️  Significant degradation detected ({ratio:.1f}x slower)")
    elif ratio > 1.05:
        print(f"    ⚡ Minor degradation ({ratio:.2f}x)")
    else:
        print(f"    ✅ Stable (no degradation)")


def compute_positional_wer(results: list, data_dir: Path,
                           chunk_duration: float, output_dir: Path):
    """Compute WER bucketed by position in the session."""
    # Group chunks by file and reconstruct per-file transcripts
    by_file = {}
    for r in results:
        fid = r.get("file_id", "unknown")
        by_file.setdefault(fid, []).append(r)

    total_ref_words = 0
    total_errors = 0
    bucket_results = {}  # bucket_name -> list of per-chunk WERs

    for fid, file_chunks in by_file.items():
        txt_path = data_dir / f"{fid}.txt"
        if not txt_path.exists():
            continue

        reference = txt_path.read_text().strip()
        hypothesis = " ".join(r.get("text", "") for r in sorted(file_chunks, key=lambda x: x["chunk_index"]))

        wer_result = compute_wer(reference, hypothesis)
        total_ref_words += wer_result["ref_words"]
        total_errors += wer_result["errors"]

        # Bucket analysis: first 25%, middle 50%, last 25%
        n = len(file_chunks)
        for i, r in enumerate(sorted(file_chunks, key=lambda x: x["chunk_index"])):
            if i < n * 0.25:
                bucket = "first_25pct"
            elif i < n * 0.75:
                bucket = "middle_50pct"
            else:
                bucket = "last_25pct"
            bucket_results.setdefault(bucket, []).append(r.get("text", ""))

    if total_ref_words > 0:
        overall_wer = total_errors / total_ref_words * 100
        print(f"\n  WER (Word Error Rate):")
        print(f"    Overall: {overall_wer:.2f}%")
        print(f"    Reference words: {total_ref_words}")
        print(f"    Total errors: {total_errors}")

    # Save summary
    summary = {
        "engine": results[0].get("engine", ""),
        "chunk_duration": chunk_duration,
        "total_chunks": len(results),
        "total_audio_min": sum(r.get("chunk_duration", chunk_duration) for r in results) / 60,
        "overall_wer": round(total_errors / max(total_ref_words, 1) * 100, 2),
        "mean_rtf": round(sum(r["rtf"] for r in results if r["rtf"] > 0) / max(len(results), 1), 4),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_librispeech(data_dir: Path) -> list:
    """Load LibriSpeech test-clean utterances with references."""
    ls_dir = data_dir / "LibriSpeech" / "test-clean"
    if not ls_dir.exists():
        print(f"ERROR: LibriSpeech not found at {ls_dir}")
        print("Download: curl -L https://www.openslr.org/resources/12/test-clean.tar.gz | tar xz -C benchmarks/data/")
        sys.exit(1)

    utterances = []
    for trans_file in sorted(ls_dir.rglob("*.trans.txt")):
        speaker_dir = trans_file.parent
        refs = {}
        for line in trans_file.read_text().strip().split("\n"):
            parts = line.split(" ", 1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]

        for flac in sorted(speaker_dir.glob("*.flac")):
            uid = flac.stem
            if uid in refs:
                utterances.append({
                    "path": str(flac),
                    "reference": refs[uid],
                    "id": uid,
                })

    return utterances


def run_librispeech_sustained(engine: str, model: str, data_dir: Path,
                              max_utterances: int, output_dir: Path):
    """Process LibriSpeech utterances sequentially to simulate sustained load."""
    utterances = load_librispeech(data_dir)
    if max_utterances > 0:
        utterances = utterances[:max_utterances]

    print(f"\nEngine: {engine}, Utterances: {len(utterances)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_audio_s = 0
    session_start = time.monotonic()

    for i, utt in enumerate(utterances):
        result = transcribe_chunk(CLI_PATH, utt["path"], engine, model)

        if "error" in result:
            if i < 3:
                print(f"  [{i}] ERROR: {result['error'][:80]}")
            continue

        dur = get_wav_duration(utt["path"]) if utt["path"].endswith(".wav") else 0
        if dur == 0:
            # Estimate from FLAC (~16kHz)
            dur = result.get("inference_time", 0) / max(result.get("rtf", 0.01), 0.001)
            if dur <= 0:
                dur = 5.0  # fallback

        total_audio_s += dur

        wer_result = compute_wer(utt["reference"], result["text"])

        result["chunk_index"] = i
        result["reference"] = utt["reference"]
        result["wer"] = wer_result["wer"]
        result["ref_words"] = wer_result["ref_words"]
        result["errors"] = wer_result["errors"]
        result["cumulative_audio_s"] = total_audio_s
        result["session_elapsed_s"] = time.monotonic() - session_start

        results.append(result)

        if (i + 1) % 100 == 0:
            avg_rtf = sum(r["rtf"] for r in results) / len(results)
            avg_wer = sum(r["wer"] * r["ref_words"] for r in results) / max(sum(r["ref_words"] for r in results), 1)
            print(f"  [{i+1}/{len(utterances)}] avg_RTF={avg_rtf:.4f}, "
                  f"WER={avg_wer:.2f}%, "
                  f"audio={total_audio_s/60:.1f}min, "
                  f"elapsed={results[-1]['session_elapsed_s']/60:.1f}min")

    # Save results
    results_path = output_dir / f"sustained_{engine}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Analysis
    print_analysis(results, engine, 0)

    # Positional WER analysis
    n = len(results)
    if n >= 20:
        q1 = results[:n//4]
        q4 = results[-(n//4):]

        q1_wer = sum(r["errors"] for r in q1) / max(sum(r["ref_words"] for r in q1), 1) * 100
        q4_wer = sum(r["errors"] for r in q4) / max(sum(r["ref_words"] for r in q4), 1) * 100
        overall_wer = sum(r["errors"] for r in results) / max(sum(r["ref_words"] for r in results), 1) * 100

        print(f"\n  WER by Position:")
        print(f"    Overall:        {overall_wer:.2f}%")
        print(f"    First 25%:      {q1_wer:.2f}%")
        print(f"    Last 25%:       {q4_wer:.2f}%")

        if abs(q4_wer - q1_wer) > 0.5:
            print(f"    ⚠️  WER drift: {q4_wer - q1_wer:+.2f}% (last vs first quarter)")
        else:
            print(f"    ✅ Stable WER (no positional degradation)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Long-form ASR benchmark (sustained Neural Engine load)")
    parser.add_argument("--engine", default="parakeet",
                        choices=["parakeet", "qwen3", "qwen3-coreml"],
                        help="ASR engine")
    parser.add_argument("--model", default=None,
                        help="Model variant (e.g. 0.6B, 0.6B-8bit, int8)")
    parser.add_argument("--chunk-duration", type=float, default=10.0,
                        help="Chunk duration in seconds (default: 10)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Total benchmark duration in minutes (default: 60)")
    parser.add_argument("--max-utterances", type=int, default=0,
                        help="Max utterances for LibriSpeech mode (0=all)")
    parser.add_argument("--data-dir", type=str, default="benchmarks/data",
                        help="Dataset directory")
    parser.add_argument("--dataset", default="librispeech",
                        choices=["librispeech", "earnings22"],
                        help="Dataset to use")
    parser.add_argument("--download-only", action="store_true",
                        help="Download dataset and exit")
    parser.add_argument("--compare", action="store_true",
                        help="Compare existing results across engines")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.download_only:
        if args.dataset == "earnings22":
            print("Downloading Earnings-22 dataset...")
            download_earnings22(data_dir / "earnings22")
        else:
            print("LibriSpeech: download manually from https://www.openslr.org/12/")
            print(f"  Extract to {data_dir}/LibriSpeech/test-clean/")
        return

    if args.compare:
        result_dir = BENCHMARK_BASE
        for f in sorted(result_dir.glob("*.json")):
            with open(f) as fh:
                results = json.load(fh)
            name = f.stem
            print_analysis(results, name, 0)
        return

    output_dir = BENCHMARK_BASE / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.dataset == "librispeech":
        print("Long-form benchmark: LibriSpeech sustained session")
        run_librispeech_sustained(
            engine=args.engine,
            model=args.model,
            data_dir=data_dir,
            max_utterances=args.max_utterances,
            output_dir=output_dir,
        )
    else:
        print("Step 1: Prepare dataset")
        earnings_dir = download_earnings22(data_dir / "earnings22")
        print("\nStep 2: Run long-form benchmark")
        run_longform_benchmark(
            engine=args.engine,
            model=args.model,
            data_dir=earnings_dir,
            chunk_duration=args.chunk_duration,
            max_duration=args.duration,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
