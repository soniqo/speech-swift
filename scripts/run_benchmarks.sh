#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

dataset_root="${BENCH_DATASET_ROOT:-$repo_root/.benchmark-datasets}"
results_root="${BENCH_RESULTS_ROOT:-$repo_root/benchmark-results}"
dataset_path=""
limit="${ASR_LIMIT:-200}"
engines="${ASR_ENGINES:-qwen3-mlx-0.6b-8bit qwen3-mlx-0.6b-4bit parakeet omnilingual-mlx-300m-4bit}"
language="${ASR_LANGUAGE:-}"
run_asr=1
vad_manifest="${VAD_MANIFEST:-}"
vad_limit="${VAD_LIMIT:-}"
vad_engines="${VAD_ENGINES:-silero-coreml silero-mlx firered pyannote}"
vad_silero_onset="${VAD_SILERO_ONSET:-0.25}"
vad_silero_offset="${VAD_SILERO_OFFSET:-0.20}"
vad_silero_min_speech="${VAD_SILERO_MIN_SPEECH:-}"
vad_silero_min_silence="${VAD_SILERO_MIN_SILENCE:-}"
diarization_manifest="${DIARIZATION_MANIFEST:-}"
diarization_limit="${DIARIZATION_LIMIT:-}"
diarization_engines="${DIARIZATION_ENGINES:-sortformer-default pyannote-mlx}"
build=1
metallib=1
download_dataset=1
force_download=0
render_dashboard=1
prepare_only=0
prune="${BENCH_PRUNE_RUNS:-90}"

usage() {
  cat <<EOF
Usage: scripts/run_benchmarks.sh [options]

Runs local ASR benchmarks and records results under benchmark-results/.

Options:
  --dataset <path>          Existing LibriSpeech-style directory or TSV manifest.
  --dataset-root <path>     Root for downloaded LibriSpeech test-clean.
                            Default: .benchmark-datasets
  --results-root <path>     Output root. Default: benchmark-results
  --limit <n>               Max utterances per engine. Default: 200
  --full                    Run the full dataset; omit asr-bench --limit.
  --engines "<ids>"         Space-separated engine IDs.
  --language <iso>          Optional ASR language hint.
  --skip-asr                Do not run ASR.
  --vad-manifest <path>     Run VAD benchmark with audio/reference manifest.
  --vad-limit <n>           Max VAD files to process.
  --vad-engines "<ids>"     Space-separated VAD engine IDs.
  --vad-silero-onset <f>    Silero onset threshold. Default: 0.25
  --vad-silero-offset <f>   Silero offset threshold. Default: 0.20
  --vad-silero-min-speech <f>
                            Silero minimum speech duration in seconds.
  --vad-silero-min-silence <f>
                            Silero minimum silence duration in seconds.
  --diarization-manifest <path>
                            Run diarization benchmark with audio/RTTM manifest.
  --diarization-limit <n>   Max diarization files to process.
  --diarization-engines "<ids>"
                            Space-separated diarization engine IDs.
  --skip-build              Do not build asr-bench first.
  --skip-metallib           Do not build MLX metallib first.
  --skip-download           Do not download LibriSpeech if dataset is missing.
  --force-download          Delete and re-download LibriSpeech test-clean.
  --no-dashboard            Do not render benchmark-results/index.html.
  --prepare-only            Build tools and prepare dataset, then stop.
  --prune <n>               Keep newest N runs in dashboard data. Default: 90
  -h, --help                Show this help.

Environment overrides:
  BENCH_DATASET_ROOT, BENCH_RESULTS_ROOT, ASR_LIMIT, ASR_ENGINES,
  ASR_LANGUAGE, VAD_MANIFEST, VAD_LIMIT, VAD_ENGINES,
  VAD_SILERO_ONSET, VAD_SILERO_OFFSET,
  VAD_SILERO_MIN_SPEECH, VAD_SILERO_MIN_SILENCE,
  DIARIZATION_MANIFEST, DIARIZATION_LIMIT, DIARIZATION_ENGINES,
  HF_DOWNLOAD_STALL_TIMEOUT
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset)
      dataset_path="${2:?missing value for --dataset}"
      shift 2
      ;;
    --dataset-root)
      dataset_root="${2:?missing value for --dataset-root}"
      shift 2
      ;;
    --results-root)
      results_root="${2:?missing value for --results-root}"
      shift 2
      ;;
    --limit)
      case "${2:?missing value for --limit}" in
        all|full)
          limit=""
          ;;
        *)
          limit="$2"
          ;;
      esac
      shift 2
      ;;
    --full)
      limit=""
      shift
      ;;
    --engines)
      engines="${2:?missing value for --engines}"
      shift 2
      ;;
    --language)
      language="${2:?missing value for --language}"
      shift 2
      ;;
    --skip-asr)
      run_asr=0
      shift
      ;;
    --vad-manifest)
      vad_manifest="${2:?missing value for --vad-manifest}"
      shift 2
      ;;
    --vad-limit)
      vad_limit="${2:?missing value for --vad-limit}"
      shift 2
      ;;
    --vad-engines)
      vad_engines="${2:?missing value for --vad-engines}"
      shift 2
      ;;
    --vad-silero-onset)
      vad_silero_onset="${2:?missing value for --vad-silero-onset}"
      shift 2
      ;;
    --vad-silero-offset)
      vad_silero_offset="${2:?missing value for --vad-silero-offset}"
      shift 2
      ;;
    --vad-silero-min-speech)
      vad_silero_min_speech="${2:?missing value for --vad-silero-min-speech}"
      shift 2
      ;;
    --vad-silero-min-silence)
      vad_silero_min_silence="${2:?missing value for --vad-silero-min-silence}"
      shift 2
      ;;
    --diarization-manifest)
      diarization_manifest="${2:?missing value for --diarization-manifest}"
      shift 2
      ;;
    --diarization-limit)
      diarization_limit="${2:?missing value for --diarization-limit}"
      shift 2
      ;;
    --diarization-engines)
      diarization_engines="${2:?missing value for --diarization-engines}"
      shift 2
      ;;
    --skip-build)
      build=0
      shift
      ;;
    --skip-metallib)
      metallib=0
      shift
      ;;
    --skip-download)
      download_dataset=0
      shift
      ;;
    --force-download)
      force_download=1
      shift
      ;;
    --no-dashboard)
      render_dashboard=0
      shift
      ;;
    --prepare-only)
      prepare_only=1
      shift
      ;;
    --prune)
      prune="${2:?missing value for --prune}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$repo_root"

if [ "$run_asr" -eq 0 ] && [ -z "$vad_manifest" ] && [ -z "$diarization_manifest" ]; then
  echo "Nothing to run: ASR is disabled and no VAD/diarization manifest was supplied." >&2
  exit 2
fi

log() {
  printf '\n==> %s\n' "$*"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

flac_count() {
  find "$1" -name '*.flac' -type f 2>/dev/null | wc -l | tr -d ' '
}

transcript_count() {
  find "$1" -name '*.trans.txt' -type f 2>/dev/null | wc -l | tr -d ' '
}

dataset_complete() {
  local root="$1"
  [ -d "$root" ] || return 1
  [ "$(flac_count "$root")" -ge 2620 ] || return 1
  [ "$(transcript_count "$root")" -ge 87 ] || return 1
}

prepare_librispeech() {
  local target="$dataset_root/LibriSpeech/test-clean"
  if [ "$force_download" -eq 1 ]; then
    rm -rf "$target"
  fi

  if dataset_complete "$target"; then
    dataset_path="$target"
    log "Using LibriSpeech test-clean at $dataset_path"
    return
  fi

  if [ -d "$target" ]; then
    echo "Removing incomplete LibriSpeech test-clean at $target" >&2
    rm -rf "$target"
  fi

  if [ "$download_dataset" -eq 0 ]; then
    echo "Dataset missing and --skip-download was set: $target" >&2
    exit 1
  fi

  mkdir -p "$dataset_root"
  local archive="$dataset_root/test-clean.tar.gz"
  local partial="$archive.partial"
  local extract_dir="$dataset_root/.extract-test-clean.$$"
  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"

  log "Downloading LibriSpeech test-clean"
  run curl --fail --location --retry 5 --retry-delay 10 \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time "${HF_DOWNLOAD_STALL_TIMEOUT:-300}" \
    --output "$partial" \
    https://www.openslr.org/resources/12/test-clean.tar.gz
  mv "$partial" "$archive"

  log "Extracting LibriSpeech test-clean"
  run tar -xzf "$archive" -C "$extract_dir"
  mkdir -p "$dataset_root/LibriSpeech"
  rm -rf "$target"
  mv "$extract_dir/LibriSpeech/test-clean" "$target"
  rm -rf "$extract_dir"

  if ! dataset_complete "$target"; then
    echo "Downloaded dataset is incomplete: flac=$(flac_count "$target") trans=$(transcript_count "$target")" >&2
    exit 1
  fi

  dataset_path="$target"
}

if [ "$build" -eq 1 ] && [ "$run_asr" -eq 1 ]; then
  log "Building asr-bench"
  run swift build -c release --disable-sandbox --product asr-bench
elif [ "$run_asr" -eq 1 ] && [ "$prepare_only" -eq 0 ] && [ ! -x ".build/release/asr-bench" ]; then
  echo ".build/release/asr-bench is missing; run without --skip-build first" >&2
  exit 1
fi

if [ "$build" -eq 1 ] && [ -n "$vad_manifest" ]; then
  log "Building vad-bench"
  run swift build -c release --disable-sandbox --product vad-bench
elif [ -n "$vad_manifest" ] && [ "$prepare_only" -eq 0 ] && [ ! -x ".build/release/vad-bench" ]; then
  echo ".build/release/vad-bench is missing; run without --skip-build first" >&2
  exit 1
fi

if [ "$build" -eq 1 ] && [ -n "$diarization_manifest" ]; then
  log "Building diarization-bench"
  run swift build -c release --disable-sandbox --product diarization-bench
elif [ -n "$diarization_manifest" ] && [ "$prepare_only" -eq 0 ] && [ ! -x ".build/release/diarization-bench" ]; then
  echo ".build/release/diarization-bench is missing; run without --skip-build first" >&2
  exit 1
fi

if [ "$metallib" -eq 1 ] && { [ "$run_asr" -eq 1 ] || [ -n "$vad_manifest" ] || [ -n "$diarization_manifest" ]; }; then
  log "Building MLX metallib"
  run ./scripts/build_mlx_metallib.sh release
fi

if [ "$run_asr" -eq 1 ]; then
  if [ -z "$dataset_path" ]; then
    prepare_librispeech
  else
    if [ ! -e "$dataset_path" ]; then
      echo "Dataset does not exist: $dataset_path" >&2
      exit 1
    fi
  fi
fi

log "Benchmark configuration"
if [ "$run_asr" -eq 1 ]; then
  echo "ASR dataset: $dataset_path"
  echo "ASR limit:   ${limit:-all}"
  echo "ASR engines: $engines"
  if [ -n "$language" ]; then
    echo "ASR language: $language"
  fi
else
  echo "ASR: disabled"
fi
if [ -n "$vad_manifest" ]; then
  echo "VAD manifest: $vad_manifest"
  echo "VAD limit:    ${vad_limit:-all}"
  echo "VAD engines:  $vad_engines"
  echo "VAD Silero:   onset=$vad_silero_onset offset=$vad_silero_offset minSpeech=${vad_silero_min_speech:-default} minSilence=${vad_silero_min_silence:-default}"
fi
if [ -n "$diarization_manifest" ]; then
  echo "Diarization manifest: $diarization_manifest"
  echo "Diarization limit:    ${diarization_limit:-all}"
  echo "Diarization engines:  $diarization_engines"
fi

if [ "$prepare_only" -eq 1 ]; then
  echo "Prepare-only requested; stopping before benchmark run."
  exit 0
fi

short_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
run_key="$(date -u +%Y%m%dT%H%M%SZ)-$short_sha"
run_dir="$results_root/runs/$run_key"
mkdir -p "$run_dir"

if [ "$run_asr" -eq 1 ]; then
  read -r -a engine_args <<< "$engines"
  bench_args=(
    ".build/release/asr-bench"
    "--dataset" "$dataset_path"
    "--engines" "${engine_args[@]}"
    "--isolated"
    "--output" "$run_dir/asr.json"
  )
  if [ -n "$limit" ]; then
    bench_args+=("--limit" "$limit")
  fi
  if [ -n "$language" ]; then
    bench_args+=("--language" "$language")
  fi

  log "Running ASR benchmark"
  "${bench_args[@]}" 2>&1 | tee "$run_dir/asr.log"
fi

if [ -n "$vad_manifest" ]; then
  read -r -a vad_engine_args <<< "$vad_engines"
  vad_args=(
    ".build/release/vad-bench"
    "--manifest" "$vad_manifest"
    "--engines" "${vad_engine_args[@]}"
    "--silero-onset" "$vad_silero_onset"
    "--silero-offset" "$vad_silero_offset"
    "--output" "$run_dir/vad.json"
  )
  if [ -n "$vad_limit" ]; then
    vad_args+=("--limit" "$vad_limit")
  fi
  if [ -n "$vad_silero_min_speech" ]; then
    vad_args+=("--silero-min-speech" "$vad_silero_min_speech")
  fi
  if [ -n "$vad_silero_min_silence" ]; then
    vad_args+=("--silero-min-silence" "$vad_silero_min_silence")
  fi

  log "Running VAD benchmark"
  "${vad_args[@]}" 2>&1 | tee "$run_dir/vad.log"
fi

if [ -n "$diarization_manifest" ]; then
  read -r -a diarization_engine_args <<< "$diarization_engines"
  diarization_args=(
    ".build/release/diarization-bench"
    "--manifest" "$diarization_manifest"
    "--engines" "${diarization_engine_args[@]}"
    "--output" "$run_dir/diarization.json"
  )
  if [ -n "$diarization_limit" ]; then
    diarization_args+=("--limit" "$diarization_limit")
  fi

  log "Running diarization benchmark"
  "${diarization_args[@]}" 2>&1 | tee "$run_dir/diarization.log"
fi

python3 - "$run_dir/metadata.json" "${dataset_path:-}" "${limit:-all}" "$engines" "$run_asr" "$vad_manifest" "${vad_limit:-all}" "$vad_engines" "$vad_silero_onset" "$vad_silero_offset" "${vad_silero_min_speech:-}" "${vad_silero_min_silence:-}" "$diarization_manifest" "${diarization_limit:-all}" "$diarization_engines" <<'PY'
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def cmd(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return ""


metadata = {
    "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": "local-script",
    "git": {
        "sha": cmd("git", "rev-parse", "HEAD"),
        "short_sha": cmd("git", "rev-parse", "--short", "HEAD"),
        "branch": cmd("git", "branch", "--show-current"),
        "dirty": bool(cmd("git", "status", "--porcelain")),
    },
    "runner": {
        "host": platform.node(),
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu": cmd("sysctl", "-n", "machdep.cpu.brand_string"),
        "macos": cmd("sw_vers", "-productVersion"),
    },
    "config": {
        "dataset": sys.argv[2] or None,
        "asr_limit": sys.argv[3],
        "asr_engines": sys.argv[4].split(),
        "run_asr": sys.argv[5] == "1",
        "vad_manifest": sys.argv[6] or None,
        "vad_limit": sys.argv[7],
        "vad_engines": sys.argv[8].split(),
        "vad_silero_onset": sys.argv[9],
        "vad_silero_offset": sys.argv[10],
        "vad_silero_min_speech": sys.argv[11] or None,
        "vad_silero_min_silence": sys.argv[12] or None,
        "diarization_manifest": sys.argv[13] or None,
        "diarization_limit": sys.argv[14],
        "diarization_engines": sys.argv[15].split(),
    },
}

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
    f.write("\n")
PY

if [ "$render_dashboard" -eq 1 ]; then
  log "Rendering benchmark dashboard"
  run python3 scripts/render_benchmark_dashboard.py \
    --runs-dir "$results_root/runs" \
    --output "$results_root/index.html" \
    --prune "$prune"
fi

log "Benchmark run recorded"
echo "Run directory: $run_dir"
if [ "$render_dashboard" -eq 1 ]; then
  echo "Dashboard:     $results_root/index.html"
fi
