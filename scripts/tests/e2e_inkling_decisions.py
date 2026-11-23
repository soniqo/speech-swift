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
parser.add_argument("--backend", choices=["mlx", "coreml"], default="mlx")
parser.add_argument("--coreml-compute-units", choices=["cpu", "gpu", "ane", "all"], default="cpu")
parser.add_argument("--timeout", type=float, help="Uses the selected backend's default when omitted.")
args = parser.parse_args()
cases = json.loads((ROOT / "scripts/tests/fixtures/inkling2-decisions.json").read_text())
results = []
for case in cases:
    image = ROOT / "Tests/Qwen3ChatTests/Fixtures/Inkling2" / case["image"] if "image" in case else None
    start = time.monotonic()
    result = example.select_action(case["request"], model_dir=args.model_dir, speech_cli=args.speech, image=image,
                                   backend=args.backend, coreml_compute_units=args.coreml_compute_units,
                                   timeout=args.timeout)
    passed = result.validated and result.action == case["expected"]
    results.append({**case, **result.__dict__, "passed": passed, "seconds": time.monotonic() - start})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": args.model_dir.name, "backend": args.backend,
        "requested_compute_units": args.coreml_compute_units if args.backend == 'coreml' else None,
        "timeout_seconds": args.timeout if args.timeout is not None else (900 if args.backend == 'coreml' else 120),
        "temperature": 0, "max_tokens": 512, "reasoning_effort": 0,
        "scope": "Eight fixed action-selection cases; no actions executed", "results": results}, indent=2) + "\n")
    print(case["id"], result.action, "passed=" + str(passed), flush=True)
if not all(case["passed"] for case in results):
    sys.exit(1)
