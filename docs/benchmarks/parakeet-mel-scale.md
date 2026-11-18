# Parakeet calibration and short-window joining regression checks

Snapshot: 2026-11-18. This follow-up corrects transcript selection when overlapping hypotheses disagree. It retains the earlier paired FFT/log-floor calibration, PCM-window preprocessing, duration handling and bounded context retries. Model weights, public APIs, encoder shapes and backend selection are unchanged. Individual recognition errors and WER regressions remain.

## Why more than one fix was necessary

- The old input was `log(4E + epsilon)`, equivalent to `log(4) + log(E + epsilon/4)`. Centering removes the constant but retains a lower effective floor. Correcting FFT power alone changes weak spectral features. Scaling power and the guard together (`0.25`, and `2^-24` to `2^-26`) preserves the useful legacy ratio, subject to Float32 rounding. The explicit internal NeMo calibration retains the canonical `2^-24` guard and separate goldens.
- Correct TDT duration-zero emissions can change recognition, not merely timestamps. In a diagnostic five-second window, same-frame emissions complete an invented “la France”; the old forced advancement prevents it. Reverting the decoder also loses legitimate tokens, so this accidental suppression is not a general correction.
- Adding overlap repairs many missing passages, but matching a later word does not establish that earlier phrases agree. The previous join retained a mixed English/French prefix before “combat” and discarded the better French prefix from an original window. Three real recordings reproduce this behavior before the repair.

## Runtime behavior

- Primary PCM windows receive independent pre-emphasis, centered STFT, log features and valid-frame normalization. Constant windows and one-frame tails normalize to zero; extra STFT padding cannot displace the first valid frame.
- Encoders limited to 500 mel frames retain original cuts and add halfway context windows for longer inputs. Empty primary windows can retry with features from at most 30 seconds of surrounding audio. Exact-zero PCM and tails shorter than the maximum token span do not trigger that retry. All inference remains sequential.
- The join compares competing phrases using mean log word confidence. It rejects an alignment that would discard a stronger conflicting primary phrase away from the cut. Stronger context can still repair uncertain original text. Fallback cuts respect a retained primary word’s span, and context is retained before a delayed first primary word. Confidence is a local heuristic, not calibrated proof of correctness.
- Confidence checks inspect only overlapping text. The repair adds no model or encoder pass. Empty windows do not erase text or advance coverage; contained shorter suffixes cannot replace a completed hypothesis. Word starts remain monotonic, times are clipped to the recording, and returned confidence uses retained token probabilities.
- Inputs within one short window and exports with larger encoder windows retain their previous decoding paths. There is no automatic gain, predictor carry, language rewriting or reference-text correction.

## Native quality comparison

Baseline means the actual pre-PR runtime. “Before repair” means the previously verified overlap/retry candidate, not the earlier calibration-only proposal. Both candidates use identical cached model files and the package’s unchanged text normalizer and WER calculation.

Each cohort has 80 English and 80 French FLEURS test recordings. Original, second and third use rows 0–79, 80–159 and 160–239 per language. All three now belong to development evidence. Fresh validation uses rows 240–319, excluded from tuning; runtime sources were frozen before evaluating it and remained unchanged. Each recording is tested at original volume and a peak of `0.001`.

### INT8-iOS-5s

| Cohort | Level | Reference words | Pre-PR baseline | Before repair | Repaired candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Original | 3,873 | 16.37% (634) | 8.42% (326) | 8.34% (323) |
| Original | -60 dBFS peak | 3,873 | 19.75% (765) | 10.69% (414) | 10.51% (407) |
| Second | Original | 3,907 | 16.23% (634) | 9.55% (373) | 9.44% (369) |
| Second | -60 dBFS peak | 3,907 | 18.45% (721) | 11.54% (451) | 11.42% (446) |
| Third | Original | 3,857 | 18.69% (721) | 9.33% (360) | 9.31% (359) |
| Third | -60 dBFS peak | 3,857 | 21.10% (814) | 12.00% (463) | 11.95% (461) |
| Fresh validation | Original | 3,874 | 18.46% (715) | 10.58% (410) | 10.56% (409) |
| Fresh validation | -60 dBFS peak | 3,874 | 21.01% (814) | 11.33% (439) | 11.31% (438) |

For historical context, #559 alone produced 21.87% quiet WER on the original cohort; the earlier paired calibration/window/duration candidate produced 18.77%, and overlap/retries reduced it to 10.69%. The repaired join now produces 10.51%. These are distinct runtime stages.

### Individual regressions

| Cohort | Original-volume clips worse than pre-PR | Quiet clips worse than pre-PR | Worst added errors versus before-repair quiet output |
| --- | ---: | ---: | ---: |
| Original | 8/160 | 12/160 | 3 |
| Second | 10/160 | 11/160 | 2 |
| Third | 10/160 | 12/160 | 6 |
| Fresh validation | 14/160 | 11/160 | 3 |

Aggregate improvement does not mean every recording improves. The largest new WER increase is a French year written as “mille huit cent quatre-vingt-neuf” instead of `1889`, adding six raw WER errors without losing that information. Scores retain this formatting penalty; other individual differences must not be assumed cosmetic. On fresh quiet validation, seven clips improve and seven worsen relative to the before-repair candidate, with one fewer error in total.

### INT8-30s

| Cohort | Level | Pre-PR baseline | Repaired candidate |
| --- | --- | ---: | ---: |
| Original | Original | 5.78% (224) | 5.09% (197) |
| Original | -60 dBFS peak | 8.21% (318) | 7.64% (296) |
| Second | Original | 5.94% (232) | 5.50% (215) |
| Second | -60 dBFS peak | 7.88% (308) | 7.45% (291) |
| Third | Original | 6.53% (252) | 6.35% (245) |
| Third | -60 dBFS peak | 8.37% (323) | 7.83% (302) |
| Fresh validation | Original | 6.25% (242) | 5.91% (229) |
| Fresh validation | -60 dBFS peak | 7.18% (278) | 6.92% (268) |

All 1,388 long-export transcripts are exactly unchanged from the before-repair candidate: 640 speech recordings at both levels, 48 noisy mixtures at both levels, and twelve nonspeech inputs. The 30-second export does not use the repaired joining path.

### Added noise and nonspeech

Forty-eight mixtures use white noise at 20/10 dB SNR and pink noise at 10 dB SNR, with 1,170 reference words per level. Attenuation preserves SNR.

| Model | Level | Pre-PR baseline | Before repair | Repaired candidate |
| --- | --- | ---: | ---: | ---: |
| INT8-iOS-5s | Original | 29.66% (347) | 14.44% (169) | 14.87% (174) |
| INT8-iOS-5s | -60 dBFS peak | 30.85% (361) | 15.56% (182) | 15.64% (183) |
| INT8-30s | Original | 10.68% (125) | 9.83% (115) | 9.83% (115) |
| INT8-30s | -60 dBFS peak | 11.54% (135) | 11.11% (130) | 11.11% (130) |

The repair slightly worsens these noisy-speech mixtures relative to the prior overlap candidate. All twelve nonspeech inputs remain empty with both exports: silence, white noise, pink noise and a 60 Hz tone at 2, 5 and 12 seconds. These constructed controls do not cover all microphone environments.

Final validation contains 4,056 newly measured native Swift transcriptions, alongside retained baseline and prior-candidate evidence. Diagnostic replay is separate from these measurements. All model runs are sequential.

## Three diagnosed joins

| Quiet recording | Pre-PR baseline | Before repair | Repaired candidate | What changed |
| --- | ---: | ---: | ---: | --- |
| French 0133 | 1 error | 3 | 0 | Keeps “suite aux croisades” and the corrected “Europe occidentale” passage |
| French 0073 | 4 | 10 | 4 | Keeps “Des milliers de personnes sont mortes au combat” rather than the English-sounding overlap |
| French 0207 | 1 | 7 | 2 | Removes the duplicated English continuation; the short-window recognizer still chooses a different pronoun |

The existing 30-second export has 0, 4 and 0 errors respectively on these quiet recordings. The last pronoun error is already present in the primary recognition hypothesis, so correct assembly cannot recover it from that hypothesis alone. Longer context is a separate accuracy option; these results do not establish a corrupt export or validate a replacement model.

A blanket primary-window preference damaged other joins, so it is excluded. A prototype that assembled acoustic representations and decoded continuously with the existing five-second export also worsened the selected stress recordings and is excluded. A dedicated buffered-context export would need separate validation with defined context geometry and decoding state; see the [NVIDIA buffered inference implementation](https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py). No model was re-exported for this repair.

## Performance tradeoff

A controlled Mac comparison uses the first 16 fresh-validation recordings per language longer than five seconds, at both levels, with three repetitions and alternating implementation order: 192 transcriptions per implementation, 576 total. One model process runs at a time. Model loading and audio I/O are excluded. RTF is elapsed transcription time divided by audio duration; lower is better.

| Implementation | Median transcription time | Median RTF | Total transcription time |
| --- | ---: | ---: | ---: |
| Pre-PR baseline | 58.8 ms | 0.00553 | 11.07 s |
| Before join repair | 86.1 ms | 0.00801 | 16.68 s |
| Repaired candidate | 88.0 ms | 0.00808 | 16.90 s |

The repaired candidate takes 1.34% more total transcription time on this selection. This is a small local measurement, not a claim of statistical significance. The existing overlap path still costs additional encoder passes compared with the pre-PR runtime. Retry feature storage remains bounded to 30 seconds; whole-model and Neural Engine memory were not measured.

## Physical iPhone verification

The frozen repaired candidate and actual pre-PR baseline completed 112 transcriptions on an iPhone 16 Pro running iOS 26.6. The selection has 22 speech recordings at both volumes plus twelve nonspeech inputs per implementation. Saved device files were retrieved and compared with every console result. This is a selected stress check, not an independent accuracy estimate.

| Selected 22 speech recordings | Pre-PR baseline | Repaired candidate |
| --- | ---: | ---: |
| Original | 13.41% (85/634) | 8.36% (53/634) |
| -60 dBFS peak | 14.51% (92/634) | 8.83% (56/634) |

All 24 nonspeech outputs are empty. Protected passages, all three repaired joins, word confidence, ordered word starts and clipped word spans pass. 111/112 transcripts match the Mac exactly; all word-error counts agree. Foreground use of the temporary iPhone app was explicitly approved. The temporary app and its copied inputs were removed after verification.

| Same 20 recordings as prior phone verification | Before join repair | Repaired candidate |
| --- | ---: | ---: |
| Original | 8.05% (48/596) | 8.39% (50/596) |
| -60 dBFS peak | 11.07% (66/596) | 8.89% (53/596) |

The selected quiet stress score improves while the selected original-volume score increases by two errors. This is a deliberately selected comparison, not a general accuracy estimate. Two of the 22 quiet device recordings remain one error worse than the pre-PR baseline; all 22 original-volume recordings are no worse than that baseline.

## Verification and limits

- All 85 focused CPU tests pass: 81 Parakeet tests and four isolated-runner tests. Eight new joining tests cover conflicting prefixes, early/late anchors, stronger context, delayed primary speech and word spans. Independent legacy/canonical feature goldens remain unchanged.
- All 24 selected real-model E2E cases pass. Three new cases protect the diagnosed French joins. Existing cases cover quiet passages, silence/noise, repeated calls, confidence, duration-zero emissions, shape adaptation, language steering, latency and absolute times across 60 seconds.
- The production target builds through the full Swift 6 package in release mode, and identical runtime sources compile into the temporary iOS app. CPU-only Parakeet tests remain included in CI; model-loading E2E suites stay excluded.
- No full package E2E matrix was run as one accumulating process. The samples are finite English/French recordings with artificial attenuation and constructed noise, not a broad multilingual field evaluation. Individual clips still worsen, and confidence is not an oracle.

## Provenance

- Pre-PR baseline: `64bcdc960d78f968ab6fe5767797c2ac0dad9434`.
- FFT correction: [#559](https://github.com/soniqo/speech-swift/pull/559), `31a3412f2f4ec5c11c308f1e6b6efda7b179a0c7`.
- Before-repair production source SHA-256: `076de658de1c5436ed96a1ae349cbe0b3f543f8413bacfcd8ab3109f1191094a`.
- Repaired production source SHA-256: `47a0e55500a0c2b9096e86988dd4844b1e01ea4e79a74bc63b547dd1262047ce`, hashing sorted Swift filenames followed by their bytes.
- CoreML models: `aufklarer/Parakeet-TDT-v3-CoreML-INT8-iOS-5s` and `aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s`.
- FLEURS Parquet revision: `168de341b3db6859a9bac1c50a2ef5e3b47647e0`, English/French test rows 0–319 per language, CC BY 4.0. Added WAV fixtures include attribution and reference text.
- Earlier canonical/Float32 diagnostics used NeMo v3.0.0 preprocessing and NVIDIA original weights at `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`; those diagnostics are separate from the new joining measurements.
- Measurements: Apple M5 Pro, macOS 26.6.2, release Swift, default ANE-preferred loading policy. Device results use iPhone 16 Pro, iOS 26.6. Raw evidence and reproduction harnesses remain outside the repository.
