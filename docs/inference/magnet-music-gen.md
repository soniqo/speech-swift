# MAGNeT Inference

CLI command: `speech compose`. Generates 30 s of 32 kHz mono music from a text
prompt.

## Quick start

```bash
# Build (release + metallib)
make build

# Generate 30 s of "happy rock"
.build/release/speech compose "happy rock" -o happy_rock.wav

# Use the medium model (1.5B) — better prompt following, slower
.build/release/speech compose "energetic EDM with synth lead" \
    --variant medium-int4 -o edm.wav

# Reproducible
.build/release/speech compose "lo-fi hip hop" --seed 42 -o lofi.wav
```

## Prompt engineering tip

Short genre tags ("happy rock") work but feel thin. Descriptive prompts that
mention **instruments + tempo + mood** noticeably improve coherence:

```bash
# Thin
.build/release/speech compose "happy rock" -o out.wav

# Richer — usually better
.build/release/speech compose \
    "energetic upbeat rock anthem with electric guitar riffs, driving drums, bass groove" \
    -o out.wav
```

In our quality sweep the richer prompt gave higher zero-crossing rate
(0.116 vs 0.093, where higher = more high-frequency detail) and zero
clipping, at the cost of about a 10% lower RMS (less hot).

## Variants

```bash
--variant small-int4   # default. 287 MB on disk, ~1.4 GB peak RSS, ~0.28× RTF
--variant small-int8   # 425 MB on disk
--variant medium-int4  # 1.5B, 1.36 GB on disk
--variant medium-int8  # 1.5B, 2.10 GB on disk
```

## Sampling controls

```bash
--temperature 3.0   # annealed linearly per stage (3.0 → 0)
--top-p 0.9         # nucleus sampling threshold
--cfg-max 10.0      # max classifier-free guidance coefficient
--cfg-min 1.0       # min CFG coefficient
--steps 20,10,10,10 # decoding iterations per codebook (stages 0..3)
--seed 42           # reproducibility
```

CFG anneals **down** as the mask schedule progresses (`mask_p` goes from 1.0
to 0.0 over each stage). Higher `cfg_max` = stronger prompt adherence, more
artifacts; lower `cfg_min` = more diversity at the end of decoding.

`--steps` totals to the number of LM forwards. Default 50 = `20+10+10+10`.
Raising the first codebook count (stage 0) is the biggest quality lever; the
last 3 codebooks are refinement.

## Programmatic use

```swift
import MAGNeTMusicGen

let model = try await MAGNeTMusicGen.fromPretrained(variant: .smallInt4)
let params = MAGNeTGenerationParams(
    decodingSteps: [20, 10, 10, 10],
    maxCfgCoef: 10.0, minCfgCoef: 1.0,
    temperature: 3.0, topP: 0.9,
    annealTemp: true, seed: 42)

let pcm: [Float] = model.generate(text: "happy rock", params: params)
// pcm.count == 30 * 32_000 == 960_000

try WAVWriter.write(samples: pcm, sampleRate: 32_000,
                    to: URL(fileURLWithPath: "happy_rock.wav"))
```

## What gets downloaded

The first run pulls three repos into `~/Library/Caches/qwen3-speech/`:

| Repo | Size | Purpose |
|---|---|---|
| `aufklarer/MAGNeT-Small-30secs-MLX-4bit` (or chosen variant) | 0.5–2.2 GB | LM weights + config |
| `t5-base` | ~880 MB | Text encoder (only the encoder half is used) |
| `mlx-community/encodec-32khz-float32` | ~250 MB | Audio decoder + RVQ codebooks |

Total: ~1.7–3.3 GB depending on variant. Subsequent runs load from cache.

## License

Upstream weights are **CC-BY-NC 4.0** — non-commercial only. Same restriction
applies to anything generated. See model card on Hugging Face for details.
