#!/usr/bin/env python3
"""Check dynamic batch/sequence reuse and eager parity for one compiled graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        batch, sequence = (int(item) for item in value.lower().split("x", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must use BxS, for example 2x16") from error
    if batch < 2 or sequence < 2:
        raise argparse.ArgumentTypeError("dynamic probe dimensions must both be at least 2")
    return batch, sequence


def _counter_value(counters, group: str, name: str) -> int | None:
    values = counters.get(group)
    if values is None:
        return None
    value = values.get(name)
    return int(value) if value is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), required=True)
    parser.add_argument("--backend", default="inductor")
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        dest="shapes",
        help="Dynamic input shape BxS; repeat to add cases.",
    )
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--max-unique-graphs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npu-codegen", choices=("dvm", "ascend-triton"))
    args = parser.parse_args()

    if args.npu_codegen and args.device != "npu":
        parser.error("--npu-codegen requires --device npu")
    if args.npu_codegen:
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = (
            "dvm" if args.npu_codegen == "dvm" else "default"
        )
    os.environ.setdefault("TORCH_LOGS", "graph_breaks,recompiles,dynamic")

    import torch

    if args.device == "npu":
        import torch_npu  # noqa: F401

    class DynamicShapeModule(torch.nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.input_projection = torch.nn.Linear(hidden_size, hidden_size * 2)
            self.output_projection = torch.nn.Linear(hidden_size * 2, hidden_size)

        def forward(self, x_BSD):
            hidden_BSF = torch.nn.functional.silu(self.input_projection(x_BSD))
            output_BSD = self.output_projection(hidden_BSF)
            return output_BSD.square().mean() + output_BSD[:, ::2].sin().mean()

    shapes = args.shapes or [(2, 8), (3, 16), (4, 24)]
    torch.manual_seed(args.seed)
    eager_model = DynamicShapeModule(args.hidden_size).to(args.device)
    compiled_model = DynamicShapeModule(args.hidden_size).to(args.device)
    compiled_model.load_state_dict(eager_model.state_dict())

    try:
        from torch._dynamo.utils import counters
    except ImportError:
        counters = None
    if counters is not None:
        counters.clear()

    compiled_callable = torch.compile(
        compiled_model,
        backend=args.backend,
        dynamic=True,
        fullgraph=True,
    )
    cases: list[dict[str, object]] = []
    all_numerics_passed = True
    for case_index, (batch, sequence) in enumerate(shapes):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed + case_index)
        input_cpu_BSD = torch.randn(
            batch,
            sequence,
            args.hidden_size,
            generator=generator,
        )
        eager_input_BSD = input_cpu_BSD.to(args.device).requires_grad_(True)
        compiled_input_BSD = input_cpu_BSD.to(args.device).requires_grad_(True)
        eager_model.zero_grad(set_to_none=True)
        compiled_model.zero_grad(set_to_none=True)

        eager_output = eager_model(eager_input_BSD)
        compiled_output = compiled_callable(compiled_input_BSD)
        eager_output.backward()
        compiled_output.backward()

        output_error = float((eager_output - compiled_output).abs().detach().cpu())
        input_grad_error = float(
            (eager_input_BSD.grad - compiled_input_BSD.grad).abs().max().detach().cpu()
        )
        parameter_grad_error = max(
            float((eager.grad - compiled.grad).abs().max().detach().cpu())
            for eager, compiled in zip(
                eager_model.parameters(), compiled_model.parameters(), strict=True
            )
        )
        case_passed = (
            output_error <= 1e-5
            and input_grad_error <= 1e-5
            and parameter_grad_error <= 1e-5
        )
        all_numerics_passed = all_numerics_passed and case_passed
        cases.append(
            {
                "shape": [batch, sequence, args.hidden_size],
                "output_abs_error": output_error,
                "input_grad_max_abs_error": input_grad_error,
                "parameter_grad_max_abs_error": parameter_grad_error,
                "passed": case_passed,
            }
        )

    unique_graphs = None
    graph_breaks = None
    if counters is not None:
        unique_graphs = _counter_value(counters, "stats", "unique_graphs")
        graph_breaks = _counter_value(counters, "graph_break", "total")
        if graph_breaks is None:
            graph_breaks = sum(int(value) for value in counters.get("graph_break", {}).values())
    graph_count_passed = unique_graphs is None or unique_graphs <= args.max_unique_graphs
    graph_break_passed = graph_breaks in (None, 0)
    passed = all_numerics_passed and graph_count_passed and graph_break_passed
    payload = {
        "schema_version": 1,
        "scope": "operator dynamic-shape probe; not end-to-end Trainer acceptance",
        "device": args.device,
        "backend": args.backend,
        "npu_codegen": args.npu_codegen,
        "dynamic": True,
        "fullgraph": True,
        "max_unique_graphs": args.max_unique_graphs,
        "observed_unique_graphs": unique_graphs,
        "observed_graph_breaks": graph_breaks,
        "cases": cases,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Dynamic-shape report: {args.output}")
    print(f"Dynamic-shape acceptance: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
