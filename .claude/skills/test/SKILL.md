---
name: test
description: Run tests. Use after code changes to validate. Arguments: unit (default, no GPU), e2e (with models), filter name, or all.
disable-model-invocation: false
argument-hint: [unit|e2e|all|FilterName]
allowed-tools: Bash
---

# Test

Run unit and/or E2E tests.

- `unit` (default): Quick tests, no model downloads. CI-safe.
- `e2e`: Full pipeline tests with real models, via `scripts/test_e2e_isolated.sh`
  — one process per E2E suite. Never use plain `swift test` for the full E2E
  suite: models accumulate in a single xctest process and can exhaust system
  memory (observed: machine reboot).
- `all`: Both unit and E2E (unit phase once, then isolated E2E suites).
- Any other argument: passed as `--filter` to swift test.

```bash
arg="${ARGUMENTS:-unit}"
case "$arg" in
  unit) swift test --skip E2E 2>&1 | tail -20 ;;
  e2e)  swift build --build-tests --disable-sandbox && ./scripts/build_mlx_metallib.sh debug && E2E_SKIP_UNIT=1 scripts/test_e2e_isolated.sh 2>&1 | tail -40 ;;
  all)  swift build --build-tests --disable-sandbox && ./scripts/build_mlx_metallib.sh debug && scripts/test_e2e_isolated.sh 2>&1 | tail -40 ;;
  *)    swift test --filter "$arg" 2>&1 | tail -20 ;;
esac
```

E2E test classes MUST be prefixed with `E2E`. Unit test classes must NOT contain `E2E`.
