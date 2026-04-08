#!/usr/bin/env python3
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "densefusion_ros"))

from densefusion_ros.densefusion_core.preprocess import (  # type: ignore[reportMissingImports]
    CameraIntrinsics,
    OBJLIST,
    OnnxDenseFusion,
    Yolo11SegOnnx,
    infer_pose_onnx,
    preprocess_rgbd,
)
from densefusion_ros.densefusion_core.geometry import quaternion_matrix  # type: ignore[reportMissingImports]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DenseFusion: GT mask vs YOLO11-seg mask")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--obj_id", type=int, default=1)
    parser.add_argument("--target_label", default="duck")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--iteration", type=int, default=4)
    parser.add_argument("--pose_onnx_path", required=True)
    parser.add_argument("--refine_onnx_path", required=True)
    parser.add_argument("--yolo_seg_onnx_path", required=True)
    parser.add_argument("--yolo_score_th", type=float, default=0.25)
    parser.add_argument("--yolo_mask_th", type=float, default=0.5)
    parser.add_argument("--output_dir", default="benchmark-ouputs/yolo11-mask-compare")
    return parser.parse_args()


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.ndim == 3:
        mask_a = mask_a[:, :, 0]
    if mask_b.ndim == 3:
        mask_b = mask_b[:, :, 0]
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def summarize(values):
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {"avg": float(np.mean(arr)), "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0.0)


def project_point(pt3, intr: CameraIntrinsics):
    x, y, z = float(pt3[0]), float(pt3[1]), float(pt3[2])
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or z <= 1e-6:
        return None
    u = int(round((x * intr.cam_fx) / z + intr.cam_cx))
    v = int(round((y * intr.cam_fy) / z + intr.cam_cy))
    return (u, v)


def draw_pose(rgb: np.ndarray, quat: np.ndarray, trans: np.ndarray, intr: CameraIntrinsics, text: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not np.all(np.isfinite(quat)) or not np.all(np.isfinite(trans)):
        cv2.putText(bgr, f"{text} (invalid pose)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return bgr
    rot = quaternion_matrix(quat)[:3, :3]
    origin = trans
    axis_len = 0.05
    x_end = origin + rot[:, 0] * axis_len
    y_end = origin + rot[:, 1] * axis_len
    z_end = origin + rot[:, 2] * axis_len
    p0, px, py, pz = project_point(origin, intr), project_point(x_end, intr), project_point(y_end, intr), project_point(z_end, intr)
    if p0 is not None:
        cv2.circle(bgr, p0, 4, (0, 255, 255), -1)
        if px is not None:
            cv2.line(bgr, p0, px, (0, 0, 255), 2)
        if py is not None:
            cv2.line(bgr, p0, py, (0, 255, 0), 2)
        if pz is not None:
            cv2.line(bgr, p0, pz, (255, 0, 0), 2)
    cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return bgr


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def run():
    args = parse_args()
    if args.obj_id not in OBJLIST:
        raise ValueError(f"obj_id must be in {OBJLIST}")
    obj_index = OBJLIST.index(args.obj_id)
    intr = CameraIntrinsics()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = OnnxDenseFusion(args.pose_onnx_path, args.refine_onnx_path)
    yolo = Yolo11SegOnnx(args.yolo_seg_onnx_path, score_th=args.yolo_score_th, mask_th=args.yolo_mask_th)

    obj_dir = Path(args.dataset_root) / "data" / f"{args.obj_id:02d}"
    test_ids = []
    with open(obj_dir / "test.txt", "r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                test_ids.append(sid)
            if len(test_ids) >= args.num_samples:
                break

    gt_total_ms, yolo_total_ms, yolo_mask_ms, mask_ious, pose_trans_l2_mm = [], [], [], [], []
    valid = 0

    for sid in test_ids:
        rgb = cv2.imread(str(obj_dir / "rgb" / f"{sid}.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(obj_dir / "depth" / f"{sid}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        gt_mask = cv2.imread(str(obj_dir / "mask" / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or gt_mask is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        gt_data = preprocess_rgbd(rgb, depth, gt_mask, obj_index, args.num_points, intr, 80, 80)
        if gt_data is None:
            continue
        gt_q, gt_t, gt_conf = infer_pose_onnx(runner, gt_data, args.iteration, args.num_points)
        t1 = time.perf_counter()
        gt_total_ms.append((t1 - t0) * 1000.0)

        t2 = time.perf_counter()
        yolo_mask = yolo.infer_mask(rgb, args.target_label)
        t3 = time.perf_counter()
        yolo_mask_ms.append((t3 - t2) * 1000.0)
        yolo_q = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
        yolo_t = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        yolo_conf = 0.0
        if yolo_mask is not None:
            yolo_data = preprocess_rgbd(rgb, depth, yolo_mask, obj_index, args.num_points, intr, 80, 80)
            if yolo_data is not None:
                yolo_q, yolo_t, yolo_conf = infer_pose_onnx(runner, yolo_data, args.iteration, args.num_points)
                t4 = time.perf_counter()
                yolo_total_ms.append((t4 - t2) * 1000.0)
                mask_ious.append(iou(gt_mask, yolo_mask))
                trans_l2 = float(np.linalg.norm(gt_t - yolo_t) * 1000.0)
                if np.isfinite(trans_l2):
                    pose_trans_l2_mm.append(trans_l2)
                valid += 1
        else:
            yolo_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

        gt_mask_vis = add_title(overlay_mask(rgb, gt_mask, color=(0, 255, 0)), "GT Mask")
        yolo_mask_vis = add_title(overlay_mask(rgb, yolo_mask, color=(0, 165, 255)), f"YOLO11 Mask ({args.target_label})")
        gt_pose_vis = add_title(draw_pose(rgb, gt_q, gt_t, intr, f"GT Pose c={gt_conf:.3f}"), "Pose from GT Mask")
        pose_title = "Pose from YOLO11 Mask" if np.all(np.isfinite(yolo_t)) else "Pose from YOLO11 Mask (not found)"
        yolo_pose_vis = add_title(draw_pose(rgb, yolo_q, yolo_t, intr, f"YOLO Pose c={yolo_conf:.3f}"), pose_title)
        row = np.hstack([gt_mask_vis, yolo_mask_vis, gt_pose_vis, yolo_pose_vis])
        cv2.imwrite(str(output_dir / f"obj{args.obj_id:02d}_{sid}_compare.png"), row)

    gt_stat = summarize(gt_total_ms)
    yolo_stat = summarize(yolo_total_ms)
    yolo_mask_stat = summarize(yolo_mask_ms)
    iou_avg = statistics.mean(mask_ious) if mask_ious else 0.0
    trans_mm_avg = statistics.mean(pose_trans_l2_mm) if pose_trans_l2_mm else 0.0

    print("=== DenseFusion YOLO11-Seg Mask Benchmark ===")
    print(f"obj_id={args.obj_id} target_label={args.target_label} requested_samples={len(test_ids)} valid_compared={valid}")
    print(f"GT mask      total ms: avg={gt_stat['avg']:.3f} p50={gt_stat['p50']:.3f} p90={gt_stat['p90']:.3f}")
    print(f"YOLO11 mask  mask  ms: avg={yolo_mask_stat['avg']:.3f} p50={yolo_mask_stat['p50']:.3f} p90={yolo_mask_stat['p90']:.3f}")
    print(f"YOLO11 mask  total ms: avg={yolo_stat['avg']:.3f} p50={yolo_stat['p50']:.3f} p90={yolo_stat['p90']:.3f}")
    print(f"Mask IoU (GT vs YOLO11): avg={iou_avg:.4f}")
    print(f"Pose translation L2 diff: avg={trans_mm_avg:.3f} mm")
    print(f"Visualization outputs: {output_dir}")


if __name__ == "__main__":
    run()
