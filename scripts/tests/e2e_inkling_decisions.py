"""Opt-in real Swift inference; records raw results outside the repository."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("inkling2_decision", ROOT / "scripts/examples/inkling2_decision.py")
example = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = example
SPEC.loader.exec_module(example)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--speech", type=Path, default=ROOT / ".build/debug/speech")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
cases = json.loads((ROOT / "scripts/tests/fixtures/inkling2-decisions.json").read_text())
results = []
for case in cases:
    image = ROOT / "Tests/Qwen3ChatTests/Fixtures/Inkling2" / case["image"] if "image" in case else None
    start = time.monotonic()
    result = example.select_action(case["request"], model_dir=args.model_dir, speech_cli=args.speech, image=image)
    passed = result.validated and result.action == case["expected"]
    results.append({**case, **result.__dict__, "passed": passed, "seconds": time.monotonic() - start})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": args.model_dir.name, "scope": "Eight fixed action-selection cases; no actions executed", "results": results}, indent=2) + "\n")
    print(case["id"], result.action, "passed=" + str(passed), flush=True)
if not all(case["passed"] for case in results):
    sys.exit(1)
