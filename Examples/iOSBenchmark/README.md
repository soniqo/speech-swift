# iOSBenchmark

On-device benchmark app for the CoreML models. Loads each model, runs a warm-up
pass, then medians 5 timed runs and reports **RTF** (wall ÷ audio, lower = faster),
LLM **tokens/s**, and peak **`phys_footprint`** memory. Results print to the console
and are written to `Documents/results.json`.

Covers Parakeet-EOU (streaming ASR + EOU), Omnilingual (multilingual ASR),
Supertonic-3 and Kokoro-82M (TTS), and FunctionGemma (LLM). Latest results:
[`docs/benchmarks/ios-coreml.md`](../../docs/benchmarks/ios-coreml.md).

## Run on a device

```bash
cd Examples/iOSBenchmark
xcodegen generate                       # project.yml -> iOSBenchmark.xcodeproj
xcodebuild -project iOSBenchmark.xcodeproj -scheme iOSBenchmark \
  -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath build -allowProvisioningUpdates build

DEV=<device-udid>                        # xcrun devicectl list devices
APP=build/Build/Products/Release-iphoneos/iOSBenchmark.app
xcrun devicectl device install app --device "$DEV" "$APP"
xcrun devicectl device process launch --console --terminate-existing --device "$DEV" \
  audio.soniqo.iOSBenchmark
xcrun devicectl device copy from --device "$DEV" \
  --domain-type appDataContainer --domain-identifier audio.soniqo.iOSBenchmark \
  --source Documents/results.json --destination ./results.json
```

Set `DEVELOPMENT_TEAM` in `project.yml` to your team. Keep the phone unlocked with
Auto-Lock off during the run (the app disables the idle timer, but it must be
foreground) — a suspended app pauses downloads and inference.

## Side-loading weights (flaky network)

First run downloads ~1.5–2.5 GB of CoreML bundles from HuggingFace. If those stall,
copy the caches from a Mac that already has them (same on-disk layout):

```bash
# Mac cache -> device app container (per model)
xcrun devicectl device copy to --device "$DEV" \
  --domain-type appDataContainer --domain-identifier audio.soniqo.iOSBenchmark \
  --source "$HOME/Library/Caches/qwen3-speech/models/aufklarer/Kokoro-82M-CoreML" \
  --destination "Library/Caches/qwen3-speech/models/aufklarer/Kokoro-82M-CoreML"
# FunctionGemma uses the swift-transformers cache:
#   source ~/Documents/huggingface/models/aufklarer/FunctionGemma-270M-CoreML-Palettize8
#   dest   Documents/huggingface/models/aufklarer/FunctionGemma-270M-CoreML-Palettize8
```
