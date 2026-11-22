"""CPU-only boundaries for the local decision example."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "examples/inkling2_decision.py"
SPEC = importlib.util.spec_from_file_location("inkling2_decision", PATH)
example = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = example
SPEC.loader.exec_module(example)


class DecisionValidationTests(unittest.TestCase):
    def test_each_allowed_action(self):
        import json
        for action in example.ACTIONS:
            with self.subTest(action=action):
                value = example.parse_proposal(json.dumps({"action": action}), image_available=True)
                self.assertTrue(value.validated)
                self.assertEqual(value.action, action)

    def test_rejects_malformed_or_ambiguous_payloads(self):
        inputs = ["", "not json", '```json\n{"action":"answer"}\n```',
                  '{"action":"answer"} trailing', '[{"action":"answer"}]',
                  '{"action":"answer","command":"rm"}', '{}', 'null',
                  '{"action":true}', '{"action":1}', '{"action":["answer"]}',
                  '{"action":"run_shell"}', '{"action":"ANSWER"}',
                  '{"action":"answer","action":"clarify"}',
                  '{"action":NaN}', '{"action":Infinity}']
        for raw in inputs:
            with self.subTest(raw=raw):
                result = example.parse_proposal(raw)
                self.assertFalse(result.validated)
                self.assertEqual(result.action, "clarify")

    def test_image_action_requires_real_input(self):
        result = example.parse_proposal('{"action":"describe_image"}')
        self.assertEqual(result.fallback_reason, "missing_image")

    def test_limits_response_size_and_nesting(self):
        self.assertEqual(example.parse_proposal(" " * 4097).fallback_reason, "oversized_response")
        self.assertFalse(example.parse_proposal("[" * 1500 + "]" * 1500).validated)

    def test_clarify_can_be_a_valid_model_choice(self):
        result = example.parse_proposal('{"action":"clarify"}')
        self.assertTrue(result.validated)
        self.assertIsNone(result.fallback_reason)


class DecisionInvocationTests(unittest.TestCase):
    def invoke(self, request="What is two plus two?", **kwargs):
        return example.select_action(request, model_dir=Path("/model"),
                                     speech_cli=Path("/speech"), **kwargs)

    @patch.object(example.subprocess, "run")
    def test_invokes_swift_with_literal_data_and_greedy_settings(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, '{"action":"answer"}', "")
        result = self.invoke('$(touch sentinel) --image /secret')
        self.assertEqual(result.action, "answer")
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["/speech", "inkling"])
        self.assertIn('$(touch sentinel) --image /secret', command[2])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(command[command.index("--temperature") + 1], "0")
        self.assertEqual(command[command.index("--max-tokens") + 1], "512")

    @patch.object(example.subprocess, "run")
    def test_fails_closed_for_inference_errors(self, run):
        for failure in [FileNotFoundError(), UnicodeError(), subprocess.TimeoutExpired("speech", 1)]:
            with self.subTest(failure=type(failure).__name__):
                run.side_effect = failure
                self.assertEqual(self.invoke().action, "clarify")
                self.assertFalse(self.invoke().validated)
        run.side_effect = None
        run.return_value = subprocess.CompletedProcess([], 1, "", "load failed")
        self.assertFalse(self.invoke().validated)

    @patch.object(example.subprocess, "run")
    def test_missing_inputs_do_not_start_inference(self, run):
        self.assertEqual(self.invoke(" ").fallback_reason, "missing_request")
        self.assertEqual(self.invoke(image=Path("/definitely-missing.png")).fallback_reason, "missing_image")
        run.assert_not_called()

    def test_arguments_select_local_paths(self):
        args = example.parser().parse_args(["hello", "--model-dir", "/model", "--speech", "/speech"])
        self.assertEqual(args.model_dir, Path("/model"))
        self.assertEqual(args.speech, Path("/speech"))


if __name__ == "__main__":
    unittest.main()
