#!/usr/bin/env python3
import argparse
import statistics
import time
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch

from inference_pipeline import (
    CameraIntrinsics,
    OBJLIST,
    build_models,
    load_rgb_depth_mask,
    preprocess_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DenseFusion PyTorch vs ONNXRuntime")
    parser.add_argument("--model", type=str, required=True, help="PoseNet .pth")
    parser.add_argument("--refine_model", type=str, required=True, help="PoseRefineNet .pth (load only)")
    parser.add_argument("--onnx_model", type=str, required=True, help="PoseNet ONNX path")
    parser.add_argument("--rgb_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--obj_id", type=int, required=True)
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    return parser.parse_args()


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.array(values), q))


def stats(values: List[float]) -> Dict[str, float]:
    return {
        "avg_ms": statistics.mean(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
    }


def benchmark_torch(
    estimator: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    warmup: int,
    runs: int,
) -> Tuple[List[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with torch.no_grad():
        for _ in range(warmup):
            estimator(inputs["img"], inputs["points"], inputs["choose"], inputs["idx"])
        times = []
        output = None
        for _ in range(runs):
            t0 = time.perf_counter()
            output = estimator(inputs["img"], inputs["points"], inputs["choose"], inputs["idx"])
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    pred_r, pred_t, pred_c, _ = output
    return times, (pred_r.cpu().numpy(), pred_t.cpu().numpy(), pred_c.cpu().numpy())


def benchmark_onnx(
    onnx_path: str,
    inputs: Dict[str, torch.Tensor],
    warmup: int,
    runs: int,
) -> Tuple[List[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {
        "img": inputs["img"].numpy().astype(np.float32),
        "points": inputs["points"].numpy().astype(np.float32),
        "choose": inputs["choose"].numpy().astype(np.int64),
        "obj": inputs["idx"].numpy().astype(np.int64),
    }
    for _ in range(warmup):
        session.run(None, ort_inputs)
    times = []
    output = None
    for _ in range(runs):
        t0 = time.perf_counter()
        output = session.run(None, ort_inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    pred_r, pred_t, pred_c, _ = output
    return times, (pred_r, pred_t, pred_c)


def main() -> None:
    args = parse_args()
    if args.obj_id not in OBJLIST:
        raise ValueError(f"--obj_id must be one of {OBJLIST}, got {args.obj_id}")

    estimator, _ = build_models(args.model, args.refine_model, args.num_points)
    rgb, depth, mask = load_rgb_depth_mask(args.rgb_path, args.depth_path, args.mask_path)
    data = preprocess_input(
        rgb=rgb,
        depth=depth,
        mask=mask,
        obj_index=OBJLIST.index(args.obj_id),
        num_points=args.num_points,
        intr=CameraIntrinsics(),
    )
    if data is None:
        raise RuntimeError("No valid points after preprocessing. Check depth/mask inputs.")

    torch_times, torch_out = benchmark_torch(estimator, data, args.warmup, args.runs)
    onnx_times, onnx_out = benchmark_onnx(args.onnx_model, data, args.warmup, args.runs)

    torch_stats = stats(torch_times)
    onnx_stats = stats(onnx_times)
    diff_r = float(np.mean(np.abs(torch_out[0] - onnx_out[0])))
    diff_t = float(np.mean(np.abs(torch_out[1] - onnx_out[1])))
    diff_c = float(np.mean(np.abs(torch_out[2] - onnx_out[2])))

    print("=== DenseFusion PoseNet Benchmark (CPU) ===")
    print(f"Runs: {args.runs}, Warmup: {args.warmup}")
    print(f"PyTorch  -> avg: {torch_stats['avg_ms']:.3f} ms | std: {torch_stats['std_ms']:.3f} | p50: {torch_stats['p50_ms']:.3f} | p90: {torch_stats['p90_ms']:.3f}")
    print(f"ONNX RT  -> avg: {onnx_stats['avg_ms']:.3f} ms | std: {onnx_stats['std_ms']:.3f} | p50: {onnx_stats['p50_ms']:.3f} | p90: {onnx_stats['p90_ms']:.3f}")
    if onnx_stats["avg_ms"] > 0:
        print(f"Speedup (PyTorch / ONNX): {torch_stats['avg_ms'] / onnx_stats['avg_ms']:.3f}x")
    print("Output abs diff:")
    print(f"  pred_r mean abs diff: {diff_r:.6e}")
    print(f"  pred_t mean abs diff: {diff_t:.6e}")
    print(f"  pred_c mean abs diff: {diff_c:.6e}")


if __name__ == "__main__":
    main()
