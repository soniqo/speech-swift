# Desktop speech plugin

This is a development package. Version 0.7.0 keeps all 34 tool definitions below
the desktop discovery limits while preserving every typed CLI argument. A local
host session with 0.7.0 exposed all 34 tools and passed readiness and a
Kokoro-to-Parakeet round trip. The complete directory-review cases still need to
be finished. See plugins/soniqo-speech/review/validation.md for results.

Soniqo Speech 0.7.0 mirrors the bundled Homebrew speech CLI through 33 separate
MCP tools: 30 operational commands and three help tools, preserving 392 CLI argument
definitions from Homebrew release 0.0.29. Nested command paths have separate
tools, including gliner classify, gliner extract, and clef decide.
get_speech_status additionally checks platform and runtime
availability and lists the command catalogue without loading models.

The native runtime runs on an Apple Silicon Mac with macOS 15 or later. The
adapter requires Node.js 22 or later. It does not require an API key or a Soniqo
account. Only local desktop conversations can use this package.

## Build and install locally

Install Node.js and, when using a package without its native runtime, speech:

```bash
brew install node speech
cd plugins/soniqo-speech
npm ci --ignore-scripts
npm run build
npm test
npm run package
python3 scripts/install-local.py ../../.build/plugins/soniqo-speech-0.7.0-local.zip
```

The archive and inspection report are written to .build/plugins/. The archive
includes the bundled MCP adapter and its JavaScript dependencies; the model
weights are downloaded separately by the speech runtime.

Root plugin.json and mcp.json are canonical. The MCP command points to the
contained executable scripts/start.sh; the desktop loader rejects absolute
executable paths. Protocol tests use the packaged launcher and transport.

To include an existing Apple Silicon release runtime and its adjacent resources:

```bash
python3 scripts/package.py \
  --native-runtime /absolute/path/to/runtime/resources \
  --runtime-version <release-version> \
  --output /absolute/path/to/soniqo-speech-local.zip
```

Use the resource directory containing the native speech executable, not a
shell wrapper. The full package contains that executable, the reviewed resource
bundles, and mlx.metallib. It rejects symlinks, unreviewed bundles, and non-system
dynamic dependencies. The CLI binary and machine-readable help must match the
frozen catalogue. speech-server is not required or bundled.

Before preparing a plugin for another runtime release, refresh that catalogue:

```bash
python3 scripts/sync-cli.py --binary /absolute/path/to/native/speech --runtime-version <release-version>
npm run build
```

A bundled runtime is preferred over Homebrew. SPEECH_CLI_BINARY overrides the
CLI path.
An external runtime upgrade does not automatically add tools to the frozen
plugin catalogue.

The installer registers the extracted package in the personal local marketplace
and preserves existing entries and backups. It rejects unsafe or ambiguous
archive paths and conflicting catalogue edits during installation. Restart the desktop app and install
Soniqo Speech from Soniqo Local. Registration does not activate a running chat.
The repository
catalog at .agents/plugins/marketplace.json points to plugins/soniqo-speech.
Alternatively, select that repository source in Plugins. Start a new local
conversation and select the plugin.

After updating an installed package, restart the desktop app and use a new local
conversation with the plugin selected. Earlier conversations can retain an old
server catalogue. Version 0.6.0 successfully exposed speech_transcribe after a
desktop reload; the tool then transcribed both the synthetic fixture and a newly
generated WAV. A readiness check alone does not establish that a command is
available: verify that the selected tool is exposed and returns its actual result.

For an extracted archive, add its folder to a personal marketplace while
preserving existing entries. Do not point a marketplace at an unbuilt source
folder: dist/server.cjs must exist.

## CLI command coverage

Tool names use speech_ followed by the CLI command with underscores, for example
speech_transcribe, speech_speak, speech_diarize, and speech_audio_translate.

| CLI commands | Capability |
| --- | --- |
| transcribe, transcribe-batch | ASR engines, model variants, streaming file transcription, batch processing |
| speak, kokoro, csm, qwen3-tts-coreml, vibevoice, vibevoice-encode-voice | TTS engines, preset voices, supported languages, reference audio, voice cloning and voice caches |
| respond, voice-chat, audio-translate, translate | Speech-to-speech, finite or bounded live voice chat, speech translation and text translation |
| align, vad, vad-stream, turn, diarize, embed-speaker, language-id | Alignment, speech activity, end-of-turn, speaker timelines, embeddings and language identification |
| denoise, restore, separate, compose, upsample | Audio processing, music separation and generation, super-resolution |
| gliner classify, gliner extract, clef decide | On-device text classification, entity extraction, and action selection |
| wake, avatar-motion, sa3-parity, help | Keyword detection, avatar motion, parity diagnostics and command help |

Each tool declares all native positional inputs and flags inside cli.
Execution controls remain under execution. Replace hyphens in argument
names with underscores. Boolean flags use booleans and repeating arguments use
arrays. Values are passed as argument arrays to a fixed executable, without a
shell. Use speech_help for the frozen release's command and option descriptions.
The native CLI remains responsible for engine-specific validation and model
behavior.

Tool metadata uses compact summaries. The desktop host limits each plugin tool
definition to 8,000 bytes and combined definitions to 64,000 bytes; oversized
definitions are hidden. The protocol regression check keeps this catalogue
below 7,500 bytes per tool and 58,000 bytes combined, including namespace text.
Use speech_help with cli.subcommands set to the selected command path for its
complete option descriptions and defaults. No arguments or operations are
removed to meet the metadata limits.

Optional execution controls provide a 1–1,800 second timeout (default 300),
an explicit working directory, and text for stdin. New outputs default to a
private job folder. Replacing an existing explicit output destination requires
execution.overwrite=true and an explicit user request.

Voice-chat microphone capture requires execution.allow_microphone=true and
max_seconds between 1 and 300. Using a selected MCP configuration additionally
requires execution.allow_external_tools=true. Set either opt-in only for the
user's explicit request. The plugin defaults nested writes to deny unless an
explicit mcp_write_policy is supplied; the standalone CLI defaults to allow.
Configured servers may perform external actions under the selected write policy.

CLI stdout is returned as text, with the final 64 KiB shown in the response.
The complete stdout is retained in the job's stdout.txt, up to a 64 MiB limit.
Generated files are returned as local paths and resources; inline resource
transfer is limited to 25 MiB. Native file streaming is collected until the CLI
process finishes, rather than becoming a live audio channel in the host.
The available engines, input formats and languages follow the CLI command.
Generated outputs are saved under
~/Library/Application Support/Soniqo Speech/jobs/ with private file permissions
and unique directory names. Users can delete those job directories when no
longer needed. SPEECH_PLUGIN_DATA_DIR can select a different data directory.

## Execution and data

Each CLI call starts the bundled speech executable directly; the adapter opens
no HTTP listener. Processes stop before the shared inference lock is released.
Model memory is released between calls. Concurrent plugin sessions use the same
machine-wide inference lock and refuse overlapping inference instead of queuing
additional model loads. Other speech applications and benchmarks must still be
scheduled separately.

Cancellation and timeouts terminate the native process group. CLI stdout and
generated artifacts remain in the local job folder until deleted. Stderr is
not saved; relevant CLI error lines can be returned on failure. The adapter has
no analytics, account service, or remote inference backend. Explicitly selected
voice-chat MCP servers have their own access and data handling.

Model files download from Hugging Face on first use and remain in the runtime's
cache. Selected inputs, recognized text, and returned audio pass through the
desktop host and become conversation content. Local inference does not imply
that conversation content remains only on the Mac.

After a hard process kill, a conservative lock may remain. Inspect
~/Library/Caches/Soniqo Speech/inference.lock/owner.json. Remove
that lock directory only after confirming that both pid and child_pid have
exited. Do not remove a lock while either process is running.

## Validation

```bash
cd plugins/soniqo-speech
npm run build
npm test
python3 -m unittest discover -s ../../Tests/SpeechPluginTests -p 'test_*.py'
npm run test:e2e
```

Unit/protocol tests require no model downloads. The separately enabled
E2ESpeechPluginCLI test uses real Kokoro and Parakeet weights and must run alone.
The E2E check covers mirrored CLI commands, generating
a known sentence, reading its audio resource, and verifying an ASR round trip.
Run model-heavy tests and benchmarks one at a time. Catalogue coverage is not
an inference-quality certification for every model.

## Public release preparation

The local archive is a development package. A valid ZIP alone does not establish
public-submission readiness. The public packager mode rejects missing listing
URLs, review cases, a demo URL, or the native runtime:

```bash
python3 scripts/package.py --submission
```

Keep the manifest's listing, review cases, release notes, availability, and
commerce declaration current. Verify the publisher's identity and local MCP
submission eligibility separately. Run the manifest's cases in the desktop host,
record actual tool use and playback, and provide a verified recording URL.
Do not use another product's privacy policy or terms.

Support contact: info@soniqo.audio.
