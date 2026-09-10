#!/usr/bin/env python3
"""Replay one captured GLM FlexAttention forward/backward without CP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import AuxRequest

import torchtitanturbo  # noqa: F401
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.glm5.dsa import build_dsa_block_mask


def _statistics(tensor: torch.Tensor) -> dict[str, object]:
    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    numel = tensor.numel()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite_count": finite_count,
        "numel": numel,
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "posinf_count": int(torch.isposinf(tensor).sum().item()),
        "neginf_count": int(torch.isneginf(tensor).sum().item()),
        "max_abs": (
            float(tensor.abs().max().item())
            if numel and finite_count == numel
            else None
        ),
    }


def replay(
    capture_directory: Path, device: torch.device, *, verbose: bool = True
) -> dict[str, object]:
    capture_directory = capture_directory.resolve()
    forward = torch.load(
        capture_directory / "forward.pt", map_location="cpu", weights_only=False
    )
    backward = torch.load(
        capture_directory / "backward.pt", map_location="cpu", weights_only=False
    )
    def input_tensor(name: str) -> torch.Tensor:
        return forward[name].to(device).detach().requires_grad_(True)

    q_QNH = input_tensor("q_QNH")
    k_KNH = input_tensor("k_KNH")
    v_KNV = input_tensor("v_KNV")
    attention_masks = forward["attention_masks"].to(device)
    topk_indices_QS = forward["topk_indices_QS"].to(device)
    block_mask = build_dsa_block_mask(
        topk_indices_QS,
        attention_masks,
        int(forward["block_size"]),
    )

    q_BNQH = q_QNH.transpose(0, 1).unsqueeze(0)
    k_BNKH = k_KNH.transpose(0, 1).unsqueeze(0)
    v_BNKV = v_KNV.transpose(0, 1).unsqueeze(0)
    output_BNQV, aux = FlexAttention.compiled_flex_attn(
        q_BNQH,
        k_BNKH,
        v_BNKV,
        score_mod=None,
        block_mask=block_mask,
        scale=forward["scale"],
        enable_gqa=False,
        return_aux=AuxRequest(lse=True),
        kernel_options={},
    )
    output_QNV = output_BNQV.squeeze(0).transpose(0, 1)
    grad_output_QNV = backward["grad_output_QNV"].to(device)
    replay_delta_QN = (
        output_QNV.detach().float() * grad_output_QNV.float()
    ).sum(dim=-1)
    output_QNV.backward(grad_output_QNV)

    captured_output = forward["output_QNV"].to(device)
    captured_delta_ref = backward.get("delta_ref_QN")
    delta_ref_max_abs_diff = None
    if captured_delta_ref is not None:
        delta_ref_max_abs_diff = float(
            (replay_delta_QN - captured_delta_ref.to(device)).abs().max().item()
        )
    result = {
        "capture_directory": str(capture_directory),
        "rank": forward["rank"],
        "module_fqn": forward["module_fqn"],
        "call_index": forward["call_index"],
        "captured_output": _statistics(captured_output),
        "replayed_output": _statistics(output_QNV),
        "output_max_abs_diff": float(
            (output_QNV.detach() - captured_output).abs().max().item()
        ),
        "lse": _statistics(aux.lse),
        "grad_output": _statistics(grad_output_QNV),
        "captured_output_max_abs_diff_at_backward": backward.get(
            "output_max_abs_diff"
        ),
        "captured_delta_ref": backward.get("delta_ref_statistics"),
        "replayed_delta_ref": _statistics(replay_delta_QN),
        "delta_ref_max_abs_diff": delta_ref_max_abs_diff,
        "dq": _statistics(q_QNH.grad),
        "dk": _statistics(k_KNH.grad),
        "dv": _statistics(v_KNV.grad),
    }
    actual_path = capture_directory / "actual_gradients.pt"
    if actual_path.is_file():
        actual = torch.load(actual_path, map_location="cpu", weights_only=False)
        replayed = {
            "dq_QNH": q_QNH.grad.detach().cpu(),
            "dk_KNH": k_KNH.grad.detach().cpu(),
            "dv_KNV": v_KNV.grad.detach().cpu(),
        }
        result["captured_actual_gradients"] = actual.get("statistics")
        result["captured_vs_replayed_gradients"] = {
            name: {
                "shape_equal": tuple(actual[name].shape) == tuple(replayed[name].shape),
                "max_abs_diff": (
                    float((actual[name] - replayed[name]).abs().max().item())
                    if tuple(actual[name].shape) == tuple(replayed[name].shape)
                    and bool(torch.isfinite(actual[name]).all().item())
                    else None
                ),
                "captured": _statistics(actual[name]),
                "replayed": _statistics(replayed[name]),
            }
            for name in ("dq_QNH", "dk_KNH", "dv_KNV")
        }
    result_path = capture_directory / "replay_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "output_QNV": output_QNV.detach().cpu(),
            "lse": aux.lse.detach().cpu(),
            "delta_ref_QN": replay_delta_QN.detach().cpu(),
            "dq_QNH": q_QNH.grad.detach().cpu(),
            "dk_KNH": k_KNH.grad.detach().cpu(),
            "dv_KNV": v_KNV.grad.detach().cpu(),
        },
        capture_directory / "replay_tensors.pt",
    )
    if verbose:
        print(json.dumps(result, indent=2), flush=True)
    print(f"Replay result: {result_path}", flush=True)
    return result


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directories", type=Path, nargs="+")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    capture_root = Path(args.capture_directories[0]).resolve().parents[1]
    summary_path = capture_root / "replay_summary.json"
    summary = {
        "status": "running",
        "requested_capture_count": len(args.capture_directories),
        "completed_capture_count": 0,
        "all_replayed_gradients_finite": True,
        "calls": [],
    }
    _write_summary(summary_path, summary)
    print(f"Replay summary initialized: {summary_path}", flush=True)
    for capture_directory in args.capture_directories:
        item = replay(capture_directory, device, verbose=not args.compact)
        summary["completed_capture_count"] += 1
        summary["all_replayed_gradients_finite"] = bool(
            summary["all_replayed_gradients_finite"]
        ) and all(
            item[name]["finite_count"] == item[name]["numel"]
            for name in ("dq", "dk", "dv")
        )
        summary["calls"].append(
            {
                "rank": item["rank"],
                "module_fqn": item["module_fqn"],
                "call_index": item["call_index"],
                "output_max_abs_diff": item["output_max_abs_diff"],
                "delta_ref_max_abs_diff": item["delta_ref_max_abs_diff"],
                "captured_actual_gradients": item.get("captured_actual_gradients"),
                "replayed_gradients": {
                    name: item[name] for name in ("dq", "dk", "dv")
                },
                "captured_vs_replayed_gradients": item.get(
                    "captured_vs_replayed_gradients"
                ),
            }
        )
        _write_summary(summary_path, summary)
        print(
            "Replay progress: "
            f"{summary['completed_capture_count']}/"
            f"{summary['requested_capture_count']}",
            flush=True,
        )
    summary["status"] = "completed"
    _write_summary(summary_path, summary)
    print(f"Replay summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
