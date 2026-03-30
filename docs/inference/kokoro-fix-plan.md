# Kokoro CoreML Fix Plan

## Root Cause
The current conversion (`scripts/convert_kokoro_coreml.py`) has architectural bugs:

### Missing decoder inputs
```python
# Current (BROKEN):
decoder(features, f0)

# Correct (from hexgrad/kokoro KModel.forward_with_tokens):
decoder(asr, F0_pred, N_pred, ref_s[:, :128])
```

### Missing alignment
```python
# Duration → alignment matrix → feature alignment
pred_dur = round(sigmoid(duration_proj(lstm(d))).sum(-1) / speed).clamp(min=1)
indices = repeat_interleave(arange(T), pred_dur)
pred_aln_trg[indices, arange(len(indices))] = 1

# Prosody features
en = d.transpose(-1,-2) @ pred_aln_trg
F0_pred, N_pred = predictor.F0Ntrain(en, s)

# Acoustic features  
asr = t_en @ pred_aln_trg
```

### Missing F0/N prediction
The prosody predictor (`predictor.F0Ntrain`) is never exported to CoreML.
Noise (N) predictions are completely absent.

## Architecture (from hexgrad/kokoro)

```
Components (82M total):
  bert: CustomAlbert (6.3M)      — phoneme context encoding
  bert_encoder: Linear (0.4M)     — BERT → duration space
  predictor: ProsodyPredictor (16.2M) — duration + F0 + N
  text_encoder: TextEncoder (5.6M) — phoneme → acoustic features  
  decoder: Decoder (53.3M)        — synthesis vocoder (iSTFTNet)
```

## Conversion Plan (3 CoreML models)

### Model 1: Duration Model
- Input: `input_ids [1, T]`, `ref_s [1, 256]`, `speed [1]`
- Pipeline: bert → bert_encoder → predictor.text_encoder → predictor.lstm → duration_proj
- Output: `pred_dur [1, T]`, `d [1, T, hidden]`, `t_en [1, hidden, T]`
- Note: alignment matrix built in Swift from pred_dur

### Model 2: Prosody Model (F0/N)
- Input: `en [1, hidden, frames]`, `s [1, 128]`
- Pipeline: predictor.F0Ntrain(en, s)
- Output: `F0_pred [1, frames*2]`, `N_pred [1, frames*2]`

### Model 3: Decoder (fixed-shape buckets)
- Input: `asr [1, 512, frames]`, `F0_pred [1, frames*2]`, `N_pred [1, frames*2]`, `ref_s [1, 128]`
- Pipeline: decoder(asr, F0, N, ref_s)
- Output: `audio [1, samples]`
- Buckets: 5s, 10s, 15s

## Swift-side alignment
```swift
// Build alignment matrix from duration predictions
var pred_aln_trg = zeros(T, frames)
var pos = 0
for i in 0..<T {
    let dur = Int(pred_dur[i])
    for j in 0..<dur { pred_aln_trg[i, pos+j] = 1 }
    pos += dur
}
// Compute acoustic features
let asr = matmul(t_en, pred_aln_trg)
let en = matmul(d, pred_aln_trg)
```

## Reference implementations
- mattmireles/kokoro-coreml (GitHub) — working 3-model approach
- FluidInference/kokoro-82m-coreml (HuggingFace) — pre-converted models
- hexgrad/kokoro (GitHub) — official PyTorch reference
