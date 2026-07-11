#!/usr/bin/env bash
# Full test suite with per-suite process isolation for E2E classes.
#
# `swift test` runs every suite in a single xctest process. Each E2E suite
# loads multi-GB model weights that are never released between suites, so a
# full run accumulates every model in RSS and can exhaust system memory.
# This runner executes the unit phase once (CI semantics: --skip E2E) and
# then each E2E class in its own process, bounding peak memory to the
# largest single suite.
#
# Usage:
#   scripts/test_e2e_isolated.sh                 # build assumed done: swift build --build-tests + metallib
#   E2E_SKIP_FILE=passed.txt scripts/test_e2e_isolated.sh   # resume: skip listed Module.Class entries
#   E2E_ONLY_FILTER=Magpie scripts/test_e2e_isolated.sh     # run only E2E classes matching a substring
#   E2E_SKIP_UNIT=1 scripts/test_e2e_isolated.sh            # skip the unit phase
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="${E2E_LOG_DIR:-.e2e-logs}"
SKIP_FILE="${E2E_SKIP_FILE:-}"
ONLY_FILTER="${E2E_ONLY_FILTER:-}"
mkdir -p "$LOG_DIR"

SUMMARY="$LOG_DIR/summary.txt"
: > "$SUMMARY"
OVERALL=0

record() { # status name detail
  printf '%-6s %-55s %s\n' "$1" "$2" "$3" | tee -a "$SUMMARY"
}

echo "== Listing tests =="
if ! swift test list --skip-build --disable-sandbox > "$LOG_DIR/all-tests.txt" 2> "$LOG_DIR/list.err"; then
  echo "swift test list failed (build tests first: swift build --build-tests --disable-sandbox && ./scripts/build_mlx_metallib.sh debug)" >&2
  cat "$LOG_DIR/list.err" >&2
  exit 2
fi
E2E_CLASSES=$(sed -E 's|/.*||' "$LOG_DIR/all-tests.txt" | sort -u | awk -F. '$2 ~ /^E2E/')

if [[ -z "$ONLY_FILTER" && "${E2E_SKIP_UNIT:-0}" != "1" ]]; then
  echo "== Unit phase (single process, --skip E2E) =="
  if swift test --skip-build --disable-sandbox --skip E2E > "$LOG_DIR/unit.log" 2>&1; then
    record PASS "unit-phase" "$(grep -cE "Test Case .* passed" "$LOG_DIR/unit.log") cases"
  else
    OVERALL=1
    record FAIL "unit-phase" "see $LOG_DIR/unit.log"
    grep -E "Test Case .* failed \(" "$LOG_DIR/unit.log" | head -20
  fi
fi

echo "== E2E phase (one process per class) =="
for MC in $E2E_CLASSES; do
  CLS="${MC#*.}"
  if [[ -n "$ONLY_FILTER" && "$MC" != *"$ONLY_FILTER"* ]]; then continue; fi
  if [[ -n "$SKIP_FILE" ]] && grep -qxF "$MC" "$SKIP_FILE" 2>/dev/null; then
    record SKIP "$MC" "(resume skip)"
    continue
  fi
  LOG="$LOG_DIR/$CLS.log"
  if swift test --skip-build --disable-sandbox --filter "$MC" > "$LOG" 2>&1; then
    N=$(grep -cE "Test Case .* passed" "$LOG")
    S=$(grep -cE "Test Case .* skipped" "$LOG" || true)
    record PASS "$MC" "$N passed, $S skipped"
  else
    OVERALL=1
    record FAIL "$MC" "see $LOG"
    grep -E "Test Case .* failed \(|error:" "$LOG" | head -10
  fi
done

echo "== Summary =="
cat "$SUMMARY"
FAILS=$(grep -c '^FAIL' "$SUMMARY" || true)
echo "Overall: $([[ $OVERALL -eq 0 ]] && echo GREEN || echo "RED ($FAILS failing suites)")"
exit $OVERALL
