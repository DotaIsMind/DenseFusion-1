#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from datasets.linemod.dataset import get_bbox, mask_to_bbox
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_from_matrix, quaternion_matrix


LOGGER = logging.getLogger("densefusion_inference")
OBJLIST = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]


@dataclass
class CameraIntrinsics:
    cam_fx: float = 572.41140
    cam_fy: float = 573.57043
    cam_cx: float = 325.26110
    cam_cy: float = 242.04899


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DenseFusion CPU inference pipeline")
    parser.add_argument("--dataset_root", type=str, default="./datasets/linemod/Linemod_preprocessed")
    parser.add_argument("--model", type=str, required=True, help="PoseNet .pth path")
    parser.add_argument("--refine_model", type=str, required=True, help="PoseRefineNet .pth path")
    parser.add_argument("--input_type", choices=["image", "video"], required=True)
    parser.add_argument("--obj_id", type=int, required=True, help="LineMOD object id (e.g. 1,2,4...)")
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--iteration", type=int, default=4)

    parser.add_argument("--rgb_path", type=str, default="", help="single RGB image path")
    parser.add_argument("--depth_path", type=str, default="", help="single depth png path")
    parser.add_argument("--mask_path", type=str, default="", help="single mask image path")

    parser.add_argument("--video_path", type=str, default="", help="RGB video path")
    parser.add_argument("--depth_dir", type=str, default="", help="depth frames directory")
    parser.add_argument("--mask_dir", type=str, default="", help="mask frames directory")
    parser.add_argument("--frame_ext", type=str, default="png", help="depth/mask frame extension")
    parser.add_argument("--frame_start_idx", type=int, default=0, help="frame number offset")
    parser.add_argument("--max_frames", type=int, default=-1, help="limit processed frames for video mode")

    parser.add_argument("--output_json", type=str, default="", help="save pose results to json")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cam_fx", type=float, default=572.41140)
    parser.add_argument("--cam_fy", type=float, default=573.57043)
    parser.add_argument("--cam_cx", type=float, default=325.26110)
    parser.add_argument("--cam_cy", type=float, default=242.04899)
    return parser.parse_args()


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.obj_id not in OBJLIST:
        raise ValueError(f"--obj_id must be one of {OBJLIST}, got {args.obj_id}")
    if args.input_type == "image":
        required = [args.rgb_path, args.depth_path, args.mask_path]
        if not all(required):
            raise ValueError("image mode requires --rgb_path --depth_path --mask_path")
    if args.input_type == "video":
        required = [args.video_path, args.depth_dir, args.mask_dir]
        if not all(required):
            raise ValueError("video mode requires --video_path --depth_dir --mask_dir")


def build_models(
    model_path: str,
    refine_model_path: str,
    num_points: int,
    num_objects: int = 13,
) -> Tuple[PoseNet, PoseRefineNet]:
    def unwrap_dataparallel(module: nn.Module) -> nn.Module:
        if isinstance(module, nn.DataParallel):
            return module.module
        for name, child in module.named_children():
            setattr(module, name, unwrap_dataparallel(child))
        return module

    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    estimator = estimator.to("cpu")
    refiner = refiner.to("cpu")

    estimator.load_state_dict(torch.load(model_path, map_location="cpu"))
    refiner.load_state_dict(torch.load(refine_model_path, map_location="cpu"))
    estimator = unwrap_dataparallel(estimator).to("cpu")
    refiner = unwrap_dataparallel(refiner).to("cpu")
    estimator.eval()
    refiner.eval()
    return estimator, refiner


def load_rgb_depth_mask(rgb_path: str, depth_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.array(Image.open(depth_path))
    mask = np.array(Image.open(mask_path))
    return rgb, depth, mask


def preprocess_input(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    obj_index: int,
    num_points: int,
    intr: CameraIntrinsics,
) -> Optional[Dict[str, torch.Tensor]]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(np.uint8)
    mask_depth = depth != 0
    mask_label = mask != 0
    valid_mask = mask_label & mask_depth

    if valid_mask.sum() == 0:
        return None

    bbox = mask_to_bbox(mask_label.astype(np.uint8))
    rmin, rmax, cmin, cmax = get_bbox(bbox)

    rgb_chw = np.transpose(rgb[:, :, :3], (2, 0, 1)).astype(np.float32)
    rgb_crop = rgb_chw[:, rmin:rmax, cmin:cmax]

    choose = valid_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if choose.size == 0:
        return None

    if choose.size > num_points:
        c_mask = np.zeros(choose.size, dtype=np.int32)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, num_points - choose.size), mode="wrap")

    xmap = np.tile(np.arange(depth.shape[0]).reshape(-1, 1), (1, depth.shape[1]))
    ymap = np.tile(np.arange(depth.shape[1]).reshape(1, -1), (depth.shape[0], 1))

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = depth_masked
    pt0 = (ymap_masked - intr.cam_cx) * pt2 / intr.cam_fx
    pt1 = (xmap_masked - intr.cam_cy) * pt2 / intr.cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1) / 1000.0

    choose = choose.reshape(1, -1).astype(np.int64)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    data = {
        "points": torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0),
        "choose": torch.from_numpy(choose).unsqueeze(0),
        "img": normalize(torch.from_numpy(rgb_crop)).unsqueeze(0),
        "idx": torch.tensor([obj_index], dtype=torch.long),
    }
    return data


@torch.no_grad()
def infer_pose(
    estimator: PoseNet,
    refiner: PoseRefineNet,
    data: Dict[str, torch.Tensor],
    iteration: int,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    t0 = time.perf_counter()
    pred_r, pred_t, pred_c, emb = estimator(data["img"], data["points"], data["choose"], data["idx"])
    t1 = time.perf_counter()

    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(1, num_points)
    max_conf, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().numpy()
    my_t = (data["points"].view(num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().numpy()

    t2 = time.perf_counter()
    for _ in range(iteration):
        t_tensor = torch.from_numpy(my_t.astype(np.float32)).view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        mat = quaternion_matrix(my_r)
        r_tensor = torch.from_numpy(mat[:3, :3].astype(np.float32)).view(1, 3, 3)
        mat[0:3, 3] = my_t

        new_points = torch.bmm((data["points"] - t_tensor), r_tensor).contiguous()
        pred_r_refine, pred_t_refine = refiner(new_points, emb, data["idx"])
        pred_r_refine = pred_r_refine.view(1, 1, -1)
        pred_r_refine = pred_r_refine / torch.norm(pred_r_refine, dim=2).view(1, 1, 1)

        my_r_2 = pred_r_refine.view(-1).cpu().numpy()
        my_t_2 = pred_t_refine.view(-1).cpu().numpy()
        mat_2 = quaternion_matrix(my_r_2)
        mat_2[0:3, 3] = my_t_2

        mat_final = np.dot(mat, mat_2)
        r_final = mat_final.copy()
        r_final[0:3, 3] = 0
        my_r = quaternion_from_matrix(r_final, True)
        my_t = np.array([mat_final[0][3], mat_final[1][3], mat_final[2][3]])
    t3 = time.perf_counter()

    timings = {
        "estimator_ms": (t1 - t0) * 1000.0,
        "refiner_ms": (t3 - t2) * 1000.0,
        "total_ms": (t3 - t0) * 1000.0,
    }
    return my_r, my_t, float(max_conf.item()), timings


def single_image_generator(args: argparse.Namespace) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    rgb, depth, mask = load_rgb_depth_mask(args.rgb_path, args.depth_path, args.mask_path)
    yield 0, rgb, depth, mask


def video_generator(args: argparse.Namespace) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_path}")

    frame_count = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_idx = args.frame_start_idx + frame_count
        name = f"{frame_idx:06d}.{args.frame_ext}"
        depth_path = os.path.join(args.depth_dir, name)
        mask_path = os.path.join(args.mask_dir, name)
        if not os.path.exists(depth_path) or not os.path.exists(mask_path):
            LOGGER.warning("Skip frame %d due to missing depth/mask: %s %s", frame_idx, depth_path, mask_path)
            frame_count += 1
            continue

        depth = np.array(Image.open(depth_path))
        mask = np.array(Image.open(mask_path))
        yield frame_idx, rgb, depth, mask

        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break
    cap.release()


def main() -> None:
    args = parse_args()
    setup_logger()
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    intr = CameraIntrinsics(
        cam_fx=args.cam_fx,
        cam_fy=args.cam_fy,
        cam_cx=args.cam_cx,
        cam_cy=args.cam_cy,
    )
    obj_index = OBJLIST.index(args.obj_id)

    LOGGER.info("Loading models on CPU...")
    estimator, refiner = build_models(args.model, args.refine_model, args.num_points)
    LOGGER.info("Models ready. input_type=%s obj_id=%d", args.input_type, args.obj_id)

    generator = single_image_generator(args) if args.input_type == "image" else video_generator(args)

    outputs: List[Dict[str, object]] = []
    for frame_id, rgb, depth, mask in generator:
        t_pre0 = time.perf_counter()
        data = preprocess_input(
            rgb=rgb,
            depth=depth,
            mask=mask,
            obj_index=obj_index,
            num_points=args.num_points,
            intr=intr,
        )
        t_pre1 = time.perf_counter()
        if data is None:
            LOGGER.warning("Frame %s skipped: no valid masked depth points.", frame_id)
            continue

        my_r, my_t, conf, timings = infer_pose(
            estimator=estimator,
            refiner=refiner,
            data=data,
            iteration=args.iteration,
            num_points=args.num_points,
        )
        preprocess_ms = (t_pre1 - t_pre0) * 1000.0
        LOGGER.info(
            "Frame %s | preprocess=%.2fms estimator=%.2fms refiner=%.2fms total=%.2fms conf=%.4f",
            frame_id,
            preprocess_ms,
            timings["estimator_ms"],
            timings["refiner_ms"],
            timings["total_ms"],
            conf,
        )

        outputs.append(
            {
                "frame_id": int(frame_id),
                "obj_id": int(args.obj_id),
                "quaternion": [float(v) for v in my_r.tolist()],
                "translation_xyz_m": [float(v) for v in my_t.tolist()],
                "confidence": conf,
                "timing_ms": {
                    "preprocess_ms": preprocess_ms,
                    "estimator_ms": timings["estimator_ms"],
                    "refiner_ms": timings["refiner_ms"],
                    "total_ms": timings["total_ms"],
                },
            }
        )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)
        LOGGER.info("Saved %d pose results to %s", len(outputs), args.output_json)
    else:
        LOGGER.info("Finished. Valid frames=%d", len(outputs))


if __name__ == "__main__":
    main()
