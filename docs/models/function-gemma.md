# FunctionGemma Model

CoreML backend for a small tool-calling LLM exposed by the `FunctionGemma`
Swift target. It is intended for structured function and tool invocation in
local voice-agent pipelines.

## Overview

`FunctionGemma` loads the published CoreML bundle:

- `aufklarer/FunctionGemma-270M-CoreML-Palettize8` (default)
- `aufklarer/FunctionGemma-270M-CoreML` (fp16 bundle, when loaded explicitly)

The default path runs on CoreML with `cpuAndNeuralEngine` compute units and a
stateful 128-token cache. It is not a general chat model: prompts are formatted
as a single user request plus a function declaration block, then the model emits
one or more `<start_function_call>...<end_function_call>` blocks.

## Runtime Shape

| Property | Value |
|---|---:|
| Parameters | 270M class |
| Hidden size | 640 |
| Layers | 18 |
| Cache length | 128 tokens |
| Head dim | 256 |
| Backend | CoreML |
| Default compute units | CPU + Neural Engine |

The runtime builds RoPE tables in Swift, performs one stateful prefill, and then
decodes greedily one token per step into the same `MLState` cache.

## Swift API

```swift
import FunctionGemma

let model = try await FunctionGemma.loadFromHub()

let weather = FunctionDeclaration(
    name: "get_weather",
    description: "Get the current weather for a city",
    parameters: [
        "type": "object",
        "properties": [
            "city": ["type": "string"]
        ],
        "required": ["city"]
    ]
)

let result = try await model.generateCalls(
    prompt: "What is the weather in Berlin?",
    tools: [weather]
)

for call in result.calls {
    print(call.name, call.arguments)
}
```

## Integration

`FunctionGemmaPipelineLLM` adapts the model to the shared `PipelineLLM` surface
used by `SpeechCore`. Keep prompts short: the CoreML cache is fixed at 128
tokens, including function declarations and generated output.

## Source Files

```text
Sources/FunctionGemma/
  FunctionGemma.swift             CoreML loader, prefill/decode loop
  FunctionGemmaTypes.swift        FunctionDeclaration, FunctionCall, ArgumentValue
  FunctionGemmaPrompt.swift       Tool declaration and prompt formatter
  FunctionGemmaParser.swift       Function-call block parser
  FunctionGemmaPipelineLLM.swift  SpeechCore PipelineLLM adapter
```
