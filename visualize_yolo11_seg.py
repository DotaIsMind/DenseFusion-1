#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "densefusion_ros"))

from densefusion_ros.densefusion_core.preprocess import Yolo11SegOnnx  # type: ignore[reportMissingImports]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO11 segmentation detections")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--obj_id", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--yolo_seg_onnx_path", required=True)
    parser.add_argument("--score_th", type=float, default=0.25)
    parser.add_argument("--output_dir", default="benchmark-ouputs/yolo11-seg-only")
    return parser.parse_args()


def color_for_label(label: str):
    seed = sum(ord(c) for c in label) % 255
    return (int((seed * 37) % 255), int((seed * 67) % 255), int((seed * 97) % 255))


def run():
    args = parse_args()
    yolo = Yolo11SegOnnx(args.yolo_seg_onnx_path, score_th=args.score_th, mask_th=0.5)
    obj_dir = Path(args.dataset_root) / "data" / f"{args.obj_id:02d}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ids = []
    with open(obj_dir / "test.txt", "r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                test_ids.append(sid)
            if len(test_ids) >= args.num_samples:
                break

    label_counter = Counter()
    per_image = []
    for sid in test_ids:
        rgb_bgr = cv2.imread(str(obj_dir / "rgb" / f"{sid}.png"), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        dets = yolo.infer_instances(rgb, score_th=args.score_th)
        vis = rgb_bgr.copy()
        for det in dets:
            label = det["label"]
            score = det["score"]
            label_counter[label] += 1
            mask = det["mask"]
            color = color_for_label(label)
            overlay = vis.copy()
            overlay[mask > 0] = color
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0.0)
            ys, xs = np.where(mask > 0)
            if xs.size > 0 and ys.size > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis,
                    f"{label}:{score:.2f}",
                    (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        cv2.putText(vis, f"frame={sid} dets={len(dets)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"obj{args.obj_id:02d}_{sid}_yolo11_seg.png"), vis)
        per_image.append({"frame_id": sid, "num_detections": len(dets), "labels": [d["label"] for d in dets]})

    summary = {
        "obj_id": args.obj_id,
        "num_samples": len(test_ids),
        "score_th": args.score_th,
        "label_counts": dict(label_counter),
        "per_image": per_image,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved visualizations: {out_dir}")
    print(f"Detected labels: {dict(label_counter)}")


if __name__ == "__main__":
    run()
