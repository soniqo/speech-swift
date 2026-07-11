# Indic-Mio TTS Model

MLX runtime for Indic-Mio, an Indic/Hindi-first text-to-speech model with
inline emotion markers and an optional MioCodec speaker embedding path.

## Overview

`IndicMioTTSModel` loads:

- `aufklarer/Indic-Mio-MLX-fp16`

The model generates MioCodec content tokens from text and then decodes them to
24 kHz speech. It is exposed through the `IndicMioTTS` Swift target and the
`speech speak --engine indic-mio` CLI engine.

## Capabilities

| Capability | Status |
|---|---|
| Hindi / Indic text input | Supported |
| Inline emotion markers | Supported: `<happy>`, `<sad>`, `<angry>`, `<disgust>`, `<fear>`, `<surprise>` |
| Voice cloning | Supported through WavLM -> MioCodec global embedding |
| Streaming synthesis | Not wired yet |
| Preset speakers | Not supported; use text markers and optional reference audio |

## CLI

```bash
speech speak "नमस्ते, आज मैं बहुत खुश हूं <happy>" \
  --engine indic-mio \
  --language hindi \
  -o indic_mio.wav
```

With a reference voice:

```bash
speech speak "यह आवाज़ उसी वक्ता जैसी होनी चाहिए <happy>" \
  --engine indic-mio \
  --voice-sample reference.wav \
  -o cloned.wav
```

## Swift API

```swift
import IndicMioTTS

let model = try await IndicMioTTSModel.fromPretrained()
let audio = try await model.generate(
    text: "नमस्ते, आज मैं बहुत खुश हूं <happy>",
    language: "hindi"
)
```

For cloned output, extract or pass a 128-float MioCodec global speaker
embedding:

```swift
let embedding = try await model.extractGlobalEmbedding(
    referenceAudio: reference,
    referenceSampleRate: 24000
)

let cloned = try await model.generate(
    text: "यह एक परीक्षण है <surprise>",
    language: "hindi",
    globalEmbedding: embedding
)
```

## Source Files

```text
Sources/IndicMioTTS/
  IndicMioTTSModel.swift  Loader, generation, WavLM speaker embedding
  IndicMioQwen3.swift     Speech-token language model
  IndicMioTokenizer.swift Tokenizer wrapper
  IndicMioPrompt.swift    Chat prompt and emotion marker validation
  MioCodec*.swift         MioCodec content/global/wave decoders
```
