#!/usr/bin/env python3
"""Test Core ML conversion of Inkling-like routed experts and loop-local state.

This is a small synthetic operator probe, not an Inkling checkpoint converter.
Requires coremltools==9.0, torch==2.7.1, numpy<2.3 on an Apple Silicon Mac.
Run separately from other model tests, e.g. --output /tmp/inkling-coreml --device cpu.
"""
import argparse
from collections import Counter
import copy
import importlib.metadata
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "gpu", "ane", "all"], default="cpu")
    parser.add_argument("--state-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--explicit-state", action="store_true", help="Test FP32 convolution history as ordinary inputs/outputs instead of native states.")
    args = parser.parse_args()
    import numpy as np
    import torch
    import coremltools as ct
    from coremltools.optimize.coreml import OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights, decompress_weights
    torch.set_num_threads(2)
    torch.manual_seed(97)
    dtype = torch.float16 if args.state_dtype == "float16" else torch.float32
    np_dtype = np.float16 if args.state_dtype == "float16" else np.float32
    device = {"cpu": ct.ComputeUnit.CPU_ONLY, "gpu": ct.ComputeUnit.CPU_AND_GPU,
              "ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL}[args.device]
    args.output.mkdir(parents=True, exist_ok=True)

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = torch.nn.Parameter(torch.randn(8, 64) * .08)
            self.correction = torch.nn.Parameter(torch.randn(8) * .01)
            for name in ["gate", "up", "down"]:
                self.register_parameter(name, torch.nn.Parameter((torch.randn(8, 64, 64) * .08).half()))
            self.conv = torch.nn.Parameter(torch.randn(4, 64) * .1)
            for i in range(3):
                self.register_buffer(f"history{i}", torch.zeros(1, 3, 64, dtype=dtype))

        def mixture(self, x):
            scores = torch.sigmoid(x.float() @ self.router.T)
            selected = torch.topk(scores + self.correction, 2, dim=-1).indices
            probabilities = torch.gather(scores, -1, selected)
            probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20) * 8
            idx = selected.reshape(2)
            inputs = x.reshape(1, 64, 1).expand(2, 64, 1)
            gate = torch.bmm(torch.index_select(self.gate, 0, idx), inputs).squeeze(-1).float()
            up = torch.bmm(torch.index_select(self.up, 0, idx), inputs).squeeze(-1).float()
            hidden = (torch.tanh(gate / 4) * 4 * torch.sigmoid(gate) * torch.tanh(up / 25) * 25).half()
            out = torch.bmm(torch.index_select(self.down, 0, idx), hidden.unsqueeze(-1)).squeeze(-1)
            return (out.float() * probabilities.reshape(2, 1)).sum(0).reshape(1, 1, 64).half()

        def forward(self, x):
            for i in range(3):
                y = self.mixture(x)
                history = getattr(self, f"history{i}")
                context = torch.cat([history.float(), y.float()], dim=1)
                convolved = (context * self.conv.unsqueeze(0)).sum(1, keepdim=True)
                history[:, :, :].copy_(context[:, 1:].to(dtype))
                x = (x.float() + (convolved + y.float()) / (5 ** .5)).half()
            return x

    report = {"scope": "Synthetic 8-expert, top-2, width-64 MoE repeated three times with separate convolution histories. No checkpoint weights, attention, or vision.",
              "device_request": args.device, "state_dtype": args.state_dtype,
              "versions": {p: importlib.metadata.version(p) for p in ["torch", "coremltools", "numpy"]}, "results": {}}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        eager = Probe().eval()
        sample = torch.randn(1, 1, 64).half() * .2
        if args.explicit_state:
            class Explicit(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x, history0, history1, history2):
                    updated = []
                    for history in [history0, history1, history2]:
                        y = self.model.mixture(x)
                        context = torch.cat([history, y.float()], dim=1)
                        convolved = (context * self.model.conv.unsqueeze(0)).sum(1, keepdim=True)
                        updated.append(context[:, 1:])
                        x = (x.float() + (convolved + y.float()) / (5 ** .5)).half()
                    return x, updated[0], updated[1], updated[2]

            reference = Explicit(eager).eval()
            initial = [torch.zeros(1, 3, 64, dtype=torch.float32) for _ in range(3)]
            traced = torch.jit.trace(reference, (sample, *initial), check_trace=False)
            converted = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
                inputs=[ct.TensorType(name="x", shape=(1, 1, 64), dtype=np.float16)] +
                       [ct.TensorType(name=f"history{i}", shape=(1, 3, 64), dtype=np.float32) for i in range(3)],
                outputs=[ct.TensorType(name="output", dtype=np.float16)] +
                        [ct.TensorType(name=f"next_history{i}", dtype=np.float32) for i in range(3)],
                compute_precision=ct.precision.FLOAT32, compute_units=device)
            path = args.output / "probe-explicit-fp32-history.mlpackage"
            converted.save(str(path))
            histories = [v.numpy().copy() for v in initial]
            expected_histories = initial
            errors = []
            with torch.no_grad():
                for step in range(4):
                    value = sample + step * .01
                    expected = reference(value, *expected_histories)
                    expected_histories = list(expected[1:])
                    result = converted.predict({"x": value.numpy(), **{f"history{i}": v for i, v in enumerate(histories)}})
                    errors.append(float(np.max(np.abs(result["output"] - expected[0].numpy()))))
                    np.testing.assert_allclose(result["output"], expected[0].numpy(), rtol=.03, atol=.003)
                    for i, target in enumerate(expected_histories):
                        histories[i] = result[f"next_history{i}"]
                        assert histories[i].dtype == np.float32
                        np.testing.assert_allclose(histories[i], target.numpy(), rtol=.03, atol=.003)
            report["state_dtype"] = "float32"
            report["state_mode"] = "explicit I/O"
            report["results"]["fp16_weights_fp32_history"] = {"passed": True, "step_max_absolute_errors": errors,
                "history_dtype": "float32", "calls": 4, "histories_per_call": 3}
            report["status"] = "operator_probe_passed"
            save()
            print(json.dumps(report, indent=2))
            return
        traced_model = copy.deepcopy(eager)
        traced = torch.jit.trace(traced_model, sample, check_trace=False)
        for name, value in traced.named_buffers():
            if name.startswith("history"):
                value.zero_()
        converted = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18,
            inputs=[ct.TensorType(name="x", shape=(1, 1, 64), dtype=np.float16)],
            outputs=[ct.TensorType(name="output", dtype=np.float16)],
            states=[ct.StateType(name=f"history{i}", wrapped_type=ct.TensorType(shape=(1, 3, 64), dtype=np_dtype)) for i in range(3)],
            compute_units=device, skip_model_load=True)
        base_path = args.output / "probe-fp16.mlpackage"
        converted.save(str(base_path))
        config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4",
            granularity="per_block", block_size=32, weight_threshold=1024))
        compressed = linear_quantize_weights(converted, config)
        compressed_path = args.output / "probe-int4.mlpackage"
        compressed.save(str(compressed_path))
        decompressed_path = args.output / "probe-int4-decompressed.mlpackage"
        decompress_weights(compressed).save(str(decompressed_path))
        for label, path in [("fp16", base_path), ("int4", compressed_path)]:
            model = ct.models.MLModel(str(path), compute_units=device)
            state = model.make_state()
            reference = copy.deepcopy(eager)
            quant_reference = ct.models.MLModel(str(decompressed_path), compute_units=device) if label == "int4" else None
            quant_state = quant_reference.make_state() if quant_reference is not None else None
            errors, source_errors, quantization_nrmse = [], [], []
            start = time.perf_counter()
            with torch.no_grad():
                for step in range(4):
                    value = sample + step * .01
                    source_expected = reference(value).numpy().astype(np.float32)
                    expected = quant_reference.predict({"x": value.numpy()}, state=quant_state)["output"].astype(np.float32) if quant_reference else source_expected
                    actual = model.predict({"x": value.numpy()}, state=state)["output"].astype(np.float32)
                    errors.append(float(np.max(np.abs(expected - actual))))
                    source_errors.append(float(np.max(np.abs(source_expected - actual))))
                    quantization_nrmse.append(float(np.linalg.norm(source_expected - actual) / np.linalg.norm(source_expected)))
                    np.testing.assert_allclose(actual, expected, rtol=.03, atol=.003)
            prediction_seconds = time.perf_counter() - start
            fresh = model.predict({"x": sample.numpy()}, state=model.make_state())["output"]
            with torch.no_grad():
                fresh_expected = quant_reference.predict({"x": sample.numpy()}, state=quant_reference.make_state())["output"] if quant_reference else copy.deepcopy(eager)(sample).numpy()
            np.testing.assert_allclose(fresh, fresh_expected, rtol=.03, atol=.003)
            plan = ct.models.compute_plan.MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=device)
            assigned = Counter()
            for function in plan.model_structure.program.functions.values():
                for op in function.block.operations:
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    assigned[type(usage.preferred_compute_device).__name__ if usage else "unknown"] += 1
            spec = model.get_spec()
            mil_ops = Counter(op.type for f in spec.mlProgram.functions.values() for b in f.block_specializations.values() for op in b.operations)
            if label == "int4" and not any(name.startswith("constexpr_") for name in mil_ops):
                raise RuntimeError("Compression did not produce any compressed-weight operations")
            if mil_ops["write_state"] != 3 or mil_ops["read_state"] != 3:
                raise RuntimeError("Expected independent read/write operations for all three loop histories")
            report["results"][label] = {"passed": True, "step_max_absolute_errors": errors,
                "original_fp16_source_max_absolute_errors": source_errors, "original_fp16_source_nrmse": quantization_nrmse,
                "parity_reference": "eager PyTorch" if label == "fp16" else "Core ML with the same INT4 weights decompressed",
                "four_predictions_seconds": prediction_seconds, "compute_plan_operation_assignments": dict(assigned),
                "mil_operations": dict(mil_ops), "package_bytes": sum(p.stat().st_size for p in path.rglob('*') if p.is_file())}
            save()
        report["status"] = "operator_probe_passed"
    except Exception as error:
        report["status"] = "probe_failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
