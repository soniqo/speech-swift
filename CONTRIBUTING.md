# Contributing to speech-swift

Thanks for your interest in contributing. speech-swift is an on-device
speech SDK for Apple Silicon (macOS / iOS), built on MLX Swift and
CoreML. This document covers what we look for in pull requests, the
workflow, and the repo conventions that aren't obvious from reading
the code.

## Before you open a PR

- **Search existing issues first.** If your idea matches an open
  issue or an existing PR, comment there instead of opening a new
  thread.
- **Scope sensibly.** One PR = one concern. Bug fix, feature, refactor,
  or docs change — pick one. Bundling makes review harder and
  bisection impossible.
- **No AI mentions.** Commit messages, PR descriptions, review
  comments, and co-author tags must not mention Claude, ChatGPT, or
  any other AI tool.

## Branches and PRs

- Work on a feature branch. Branch names loosely follow
  `feat/<short-name>`, `fix/<short-name>`, `chore/<short-name>`, or
  `docs/<short-name>`.
- Rebase onto `main` before opening a PR. Do not merge `main` into
  your branch — we prefer a linear history.
- Every change lands via PR. We don't commit directly to `main`, and
  we never force-push to `main`. See
  [`CLAUDE.md`](CLAUDE.md) for the full workflow rules.
- CI must be green. The `build-and-test` job runs unit tests on
  macOS-15; the `homebrew-lint` job runs `brew style` + `brew audit`
  on the formula. Both need to pass.

## Testing

**Every new feature, model, or module MUST ship with tests.** Per the
testing guideline in `CLAUDE.md`:

- **Unit tests** for config parsing, data structures, weight loading,
  math / DSP logic — no GPU or model download needed. Place them in
  `Tests/<ModuleName>Tests/`.
- **E2E tests** for full-pipeline verification with real weights.
  Prefix the test class name with `E2E` (e.g. `E2ETranscriptionTests`)
  so CI's `--skip E2E` filter excludes them from the PR-time run.
  Running `make test` locally exercises them after the model cache
  warms up.
- **Regression tests** for bug fixes. When you fix a bug, add a test
  that would have caught it.

**What to test per change category:**

| Change | Required tests |
|---|---|
| New model / module | Unit (config, weight loading) + E2E (inference produces correct output) |
| New CLI command | Unit (argument parsing) + E2E (end-to-end with real files) |
| Bug fix | Regression test reproducing the bug |
| New protocol / type | Unit test for conformance + behaviour |
| DSP / audio processing | Unit test with known input / output pairs (byte-close or exact) |

## Module conventions

The module layout is flat — every model or capability lives in
`Sources/<ModuleName>/` and depends only on `AudioCommon` (and
`MLXCommon` if it uses MLX). Model targets never import each other.

Adding a new module:

1. Create `Sources/NewModule/`.
2. Add a `.library` + `.target` entry in `Package.swift`.
3. Conform to the relevant protocol in
   `Sources/AudioCommon/Protocols.swift`
   (`SpeechRecognitionModel`, `SpeechGenerationModel`,
   `VoiceActivityDetectionModel`, etc.).
4. Implement `ModelMemoryManageable` in a `+Memory.swift` file so
   callers can `unload()` and see `memoryFootprint`.
5. Write unit tests in `Tests/NewModuleTests/`.
6. Expose a CLI subcommand in `Sources/AudioCLILib/` if it's
   user-runnable.
7. Document the module: add a `docs/models/<module>.md` and / or
   `docs/inference/<module>.md`, and reference it in the README.

## Documentation requirements

Two documentation surfaces exist — keep them in sync when code
changes:

- **Local docs** in `docs/` (architecture, inference pipelines,
  benchmarks, shared protocols).
- **Website docs** in the separate `soniqo-web` repository that
  backs [soniqo.audio](https://soniqo.audio).

Any code change that affects:

- CLI flags → update the inference doc AND `/cli/` on soniqo-web.
- New modules / models → new `docs/models/*.md`, landing-page feature
  card on soniqo-web, and a dedicated `/guides/<module>/` page.
- Public API (protocols, types, function signatures) →
  `docs/shared-protocols.md` + `/api/` on soniqo-web.
- Performance characteristics → `docs/benchmarks/` +
  `/benchmarks/` on soniqo-web.

### Translations

`README.md` has 9 translations (`README_zh.md`, `README_ja.md`,
`README_ko.md`, `README_es.md`, `README_de.md`, `README_fr.md`,
`README_hi.md`, `README_pt.md`, `README_ru.md`). **Every README.md
change must update all 9.** No exceptions.

Website docs follow the same translation convention — every page
under `public/` has 9 mirrors under `public/{zh,ja,ko,es,de,fr,hi,pt,ru}/`.

## Commits

- Descriptive, imperative subject: `"Add X"`, `"Fix Y"`, `"Refactor Z"`.
- Body explains *why*, not *what* (the diff shows *what*).
- Reference the issue: `Resolves #123` or `Fixes #456`.
- No Claude / AI mentions. No `Co-Authored-By` for AI tools.
- If a test was added specifically because something regressed,
  mention that in the commit body.

## Build

```bash
make build              # release build + MLX metallib
make debug              # debug build
make test               # full test suite (runs E2E with model downloads)
make clean              # nuke .build
```

The `build_mlx_metallib.sh` step is critical — without it, inference
runs ~5× slower due to JIT shader compilation.

## Release flow

Tagged releases go through `.github/workflows/release.yml`:

1. Land your changes on `main` via PR.
2. Run `gh release create vX.Y.Z --target main --title vX.Y.Z --notes "..."`.
3. The release workflow:
   - Builds `audio` + `audio-server` in release mode.
   - Tars both plus `mlx.metallib` into `audio-macos-arm64.tar.gz`.
   - Uploads the tarball as a release asset.
   - Auto-bumps `Formula/speech.rb`'s `url` and `sha256` and commits
     the bump back to `main`.
4. Users upgrade via `brew update && brew upgrade speech`.

## Protocol and architecture reference

- **Protocols** — see
  [`docs/shared-protocols.md`](docs/shared-protocols.md) for the
  `SpeechRecognitionModel` / `SpeechGenerationModel` /
  `VoiceActivityDetectionModel` / `WakeWordProvider` / etc. surface
  each module conforms to.
- **Error handling** — use `AudioModelError` from `AudioCommon` for
  cross-module error reporting. Module-specific error enums are fine
  for domain details.
- **Logging** — use `AudioLog.modelLoading` / `AudioLog.inference` /
  `AudioLog.download`. No direct `print(...)` from library code;
  prints in CLI commands are fine.
- **Thread safety** — document-only. Every public model class is
  single-threaded by contract. Do not add locks or actors.
- **Memory** — CoreML models allocate on `libexec` load; use
  `unload()` via `ModelMemoryManageable` when done.

## Reporting security issues

If you find a security issue (unsafe weight loading, path traversal,
RCE via model downloads, etc.), email ivan.aufkl@gmail.com privately
before opening a public issue.

## License

By contributing, you agree your contributions will be licensed under
the same Apache-2.0 license as the rest of the project.
