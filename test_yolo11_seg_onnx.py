#!/usr/bin/env python3

"""
python3 "/home/ubuntu/tengf/vision-grab/DenseFusion-1/test_yolo11_seg_onnx.py" \
  --onnx_model "/home/ubuntu/tengf/vision-grab/DenseFusion-1/yolo11-seg-model/yolo26n-seg.onnx" \
  --rgb_dir "/home/ubuntu/tengf/vision-grab/DenseFusion-1/densefusion_ros/Linemod_preprocessed/data/01/rgb" \
  --num_samples 50 \
  --output_dir "/home/ubuntu/tengf/vision-grab/DenseFusion-1/benchmark-ouputs/yolo11-seg-test"
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "densefusion_ros"))

from densefusion_ros.densefusion_core.preprocess import Yolo11SegOnnx  # type: ignore[reportMissingImports]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO11-seg ONNX on an RGB image directory")
    parser.add_argument(
        "--onnx_model",
        default=str(THIS_DIR / "yolo11-seg-model" / "yolo26n-seg.onnx"),
        help="Path to YOLO11-seg ONNX model",
    )
    parser.add_argument(
        "--rgb_dir",
        default=str(THIS_DIR / "densefusion_ros" / "Linemod_preprocessed" / "data" / "01" / "rgb"),
        help="Directory of RGB images",
    )
    parser.add_argument("--score_th", type=float, default=0.25, help="Detection score threshold")
    parser.add_argument("--mask_th", type=float, default=0.5, help="Mask binarization threshold")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of images to test")
    parser.add_argument("--output_dir", default="benchmark-ouputs/yolo11-seg-test", help="Output directory")
    return parser.parse_args()


def list_images(rgb_dir: Path) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    image_paths: List[Path] = []
    for ext in exts:
        image_paths.extend(sorted(rgb_dir.glob(ext)))
    return sorted(image_paths)


def color_for_label(label: str) -> tuple[int, int, int]:
    seed = sum(ord(ch) for ch in label) % 255
    return (int((seed * 37) % 255), int((seed * 67) % 255), int((seed * 97) % 255))


def draw_detections(vis_bgr: np.ndarray, detections: Iterable[dict]) -> np.ndarray:
    canvas = vis_bgr.copy()
    for det in detections:
        label = str(det["label"])
        score = float(det["score"])
        mask = det["mask"]
        color = color_for_label(label)

        overlay = canvas.copy()
        overlay[mask > 0] = color
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0)

        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{label}:{score:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return canvas


def main() -> None:
    args = parse_args()
    rgb_dir = Path(args.rgb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not Path(args.onnx_model).is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_model}")

    image_paths = list_images(rgb_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {rgb_dir}")
    if args.num_samples > 0:
        image_paths = image_paths[: args.num_samples]

    yolo = Yolo11SegOnnx(args.onnx_model, score_th=args.score_th, mask_th=args.mask_th)

    label_counter = Counter()
    per_image = []
    infer_ms = []

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        detections = yolo.infer_instances(rgb, score_th=args.score_th)
        infer_ms.append((time.perf_counter() - t0) * 1000.0)

        for det in detections:
            label_counter[str(det["label"])] += 1

        vis = draw_detections(bgr, detections)
        cv2.putText(
            vis,
            f"{img_path.name} | dets={len(detections)}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        out_name = f"{img_path.stem}_yolo11_seg_vis.png"
        cv2.imwrite(str(output_dir / out_name), vis)

        per_image.append(
            {
                "image": img_path.name,
                "num_detections": len(detections),
                "labels": [str(d["label"]) for d in detections],
            }
        )

    avg_ms = float(np.mean(np.array(infer_ms, dtype=np.float32))) if infer_ms else 0.0
    p50_ms = float(np.percentile(np.array(infer_ms, dtype=np.float32), 50)) if infer_ms else 0.0
    p90_ms = float(np.percentile(np.array(infer_ms, dtype=np.float32), 90)) if infer_ms else 0.0

    summary = {
        "onnx_model": args.onnx_model,
        "rgb_dir": str(rgb_dir),
        "num_images": len(per_image),
        "score_th": args.score_th,
        "mask_th": args.mask_th,
        "avg_infer_ms": avg_ms,
        "p50_infer_ms": p50_ms,
        "p90_infer_ms": p90_ms,
        "label_counts": dict(label_counter),
        "per_image": per_image,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== YOLO11-Seg ONNX Test Finished ===")
    print(f"Model: {args.onnx_model}")
    print(f"RGB dir: {rgb_dir}")
    print(f"Processed images: {len(per_image)}")
    print(f"Inference ms: avg={avg_ms:.3f} p50={p50_ms:.3f} p90={p90_ms:.3f}")
    print(f"Detected labels: {dict(label_counter)}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
