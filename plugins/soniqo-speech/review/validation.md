# Desktop package validation

Version: 0.7.0. Development package; public submission still requires the final
desktop evaluation and publication materials.

## Automated evidence

- 27 Node unit/protocol checks and 21 Python installer/package checks pass without
  model downloads. The MCP client discovers all 33 command tools and the setup
  check, including nested GLiNER and Clef operations.
- The separately enabled native E2E test generates the known sentence with
  Kokoro, reads the WAV resource, and transcribes it with Parakeet. It checks
  output permissions, the audio format, actual recognized words, and lock cleanup.
- The native checks use release 0.0.29 and cached model files. A sandbox wrapper
  denies access to Homebrew's speech runtime during the packaged-runtime test.
- Binary and complete machine-readable help checksums pin the 392-argument
  catalogue. Packaging rejects symlinks, unexpected resource bundles, and
  non-system dynamic dependencies.
- A regression test reproduces the earlier discovery failure: the old speech_speak
  definition exceeded 18,000 bytes. Compact summaries preserve all 392 typed
  arguments and keep the 34 definitions at 55,684 bytes combined, with per-tool
  and combined headroom below the desktop's 8,000/64,000-byte limits. Full help
  and native defaults remain available through speech_help.
- Regression checks cover argument injection, cancellation and descendant cleanup,
  overlapping inference, derived batch-output replacement, resource traversal,
  unsafe ZIP paths, installation backups, and concurrent catalogue edits.
- The synthetic reference is maintained in the test fixtures and copied to
  review/fixtures/known-speech.wav in the archive. It says: The quick brown fox
  jumps over the lazy dog.

These checks establish the adapter and native pipeline behavior. They do not
certify every model, live microphone capture, configured external providers,
or host audio playback.

## Desktop evaluation

On 2026-11-06, a local host session using the installed 0.7.0 package exposed all
34 tools, including speech_speak and speech_transcribe_batch. Readiness reported
runtime 0.0.29. A direct plugin call generated the known sentence with Kokoro
(af_heart, English), then passed its returned WAV path to Parakeet. Both calls
returned exit code 0, and the transcript matched the sentence exactly. This
confirms host discovery and the audio workflow; it does not complete the unrun
directory-review cases or verify playback.

The local desktop host version 26.1029.70936 was tested with the plugin installed
and selected in local Work conversations. The results below are from package
0.6.0. A reload and fresh conversations resolved the earlier transcription
discovery blocker. No native transcription engine change was required. Complete
the remaining review cases with the final package before public submission.

| Case | Observed result |
| --- | --- |
| Readiness | Passed: runtime 0.0.29, 33 catalogue entries, ready status. |
| Kokoro narration | Passed: the requested sentence produced a local WAV. Playback was not requested or verified. |
| Selected-file Parakeet transcription | Passed with 0.6.0 in a fresh local conversation: speech_transcribe used Parakeet and returned the exact fixture sentence, with exit code 0. |
| Alternate preset voice | Not run in the host. |
| Combined narration/transcription | Passed with 0.6.0 in a second local conversation: Kokoro produced a 24 kHz WAV, Parakeet transcribed that returned path exactly, and the host returned a local WAV link. |
| Three negative cases | Not run in the host. |

The 0.6.0 desktop inventory contained 23 of 34 tools. The host hides definitions
that exceed either its per-tool or combined metadata budget. Version 0.7.0 fixes
that size regression; the subsequent local host session discovered all 34 tools.
Earlier conversations also retained server names from older development packages.
Restart the desktop app after updating, start a new local conversation, and
select the plugin before checking command availability. Do not mark the package
production-ready or substitute shell execution for a missing plugin tool.

The two successful desktop cases used synthetic speech containing: The quick
brown fox jumps over the lazy dog. Playback was not requested. The local runtime
performed the inference; no alternative transcription tools or shell commands
were used in these conversations.

Complete the manifest's five positive and three negative cases after discovery
is resolved. Record selected tools, actual inputs/results, errors, confirmation
behavior, artifact access, and audio playback. Rerun affected cases after
metadata or schema changes.

## Recording and publication

Capture a real, successful desktop walkthrough after the complete workflow
passes. Keep unrelated windows, private recordings, and credentials out of the
capture. Review the recording before supplying a reviewer-accessible URL.

The intended publisher is Soniqo, Inc.; verified identity and local MCP submission
eligibility remain unchecked. Availability is all supported countries. There
are no payments or purchases. Support contact: info@soniqo.audio.

The package omits unverified website, support, privacy, terms, and recording
URLs. Required portal checks and policy attestations remain outstanding. Do not
substitute another product's privacy policy or terms.
