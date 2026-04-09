#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import scipy.io as scio
import torch
import torch.nn as nn

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_from_matrix, quaternion_matrix


@dataclass
class CameraIntrinsics:
    cam_fx: float
    cam_fy: float
    cam_cx: float
    cam_cy: float


CAM_1 = CameraIntrinsics(cam_fx=1066.778, cam_fy=1067.487, cam_cx=312.9869, cam_cy=241.3109)
CAM_2 = CameraIntrinsics(cam_fx=1077.836, cam_fy=1078.189, cam_cx=323.7872, cam_cy=279.6921)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DenseFusion YCB: PyTorch vs ONNX")
    parser.add_argument("--dataset_root", default="", help="YCB_Video_Dataset root path")
    parser.add_argument("--test_list", default="datasets/ycb/dataset_config/test_data_list.txt")
    parser.add_argument(
        "--bop_scene_dir",
        default="",
        help="BOP-style scene dir, e.g. datasets/ycb-test-data/test~left_pbr/000049",
    )
    parser.add_argument("--torch_model", default="trained_checkpoints/linemod/trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth")
    parser.add_argument(
        "--torch_refine_model",
        default="trained_checkpoints/linemod/trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth",
    )
    parser.add_argument("--onnx_pose_model", default="ycb-data-onnx-model/densefusion_ycb_posenet.onnx")
    parser.add_argument("--onnx_refine_model", default="ycb-data-onnx-model/densefusion_ycb_refiner.onnx")
    parser.add_argument("--num_samples", type=int, default=100, help="Max number of frames to scan")
    parser.add_argument("--obj_id", type=int, default=0, help="1..21, or 0 for all objects in each frame")
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--input_h", type=int, default=80)
    parser.add_argument("--input_w", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup instances per backend")
    parser.add_argument("--output_json", default="", help="Optional json output path")
    return parser.parse_args()


def unwrap_dataparallel(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.DataParallel):
        return module.module
    for name, child in module.named_children():
        setattr(module, name, unwrap_dataparallel(child))
    return module


def build_torch_models(model_path: str, refine_path: str, num_points: int, num_obj: int) -> Tuple[PoseNet, PoseRefineNet]:
    estimator = PoseNet(num_points=num_points, num_obj=num_obj).to("cpu")
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj).to("cpu")
    estimator.load_state_dict(torch.load(model_path, map_location="cpu"))
    refiner.load_state_dict(torch.load(refine_path, map_location="cpu"))
    estimator = unwrap_dataparallel(estimator).to("cpu").eval()
    refiner = unwrap_dataparallel(refiner).to("cpu").eval()
    return estimator, refiner


def load_test_ids(test_list_path: str, num_samples: int) -> List[str]:
    ids: List[str] = []
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if sid:
                ids.append(sid)
            if len(ids) >= num_samples:
                break
    return ids


def load_bop_scene_data(scene_dir: Path):
    with open(scene_dir / "scene_camera.json", "r", encoding="utf-8") as f:
        scene_camera = json.load(f)
    with open(scene_dir / "scene_gt.json", "r", encoding="utf-8") as f:
        scene_gt = json.load(f)
    frame_ids = sorted(scene_gt.keys(), key=lambda x: int(x))
    return scene_camera, scene_gt, frame_ids


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {"avg": float(np.mean(arr)), "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}


def get_intrinsics_for_sid(sid: str) -> CameraIntrinsics:
    # YCB train/eval pipeline switches camera intrinsics after sequence 60.
    if not sid.startswith("data_syn") and len(sid) >= 9:
        seq = int(sid[5:9])
        if seq >= 60:
            return CAM_2
    return CAM_1


def get_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x = y = w = h = 0
    for contour in contours:
        tx, ty, tw, th = cv2.boundingRect(contour)
        if tw * th > w * h:
            x, y, w, h = tx, ty, tw, th
    if w <= 0 or h <= 0:
        return None
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    bbx = [y, y + h, x, x + w]
    bbx[0] = max(bbx[0], 0)
    bbx[1] = min(bbx[1], 479)
    bbx[2] = max(bbx[2], 0)
    bbx[3] = min(bbx[3], 639)
    rmin, rmax, cmin, cmax = bbx
    r_b = rmax - rmin
    c_b = cmax - cmin
    for idx in range(len(border_list) - 1):
        if border_list[idx] < r_b < border_list[idx + 1]:
            r_b = border_list[idx + 1]
            break
    for idx in range(len(border_list) - 1):
        if border_list[idx] < c_b < border_list[idx + 1]:
            c_b = border_list[idx + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        rmax += -rmin
        rmin = 0
    if cmin < 0:
        cmax += -cmin
        cmin = 0
    if rmax > 480:
        rmin -= rmax - 480
        rmax = 480
    if cmax > 640:
        cmin -= cmax - 640
        cmax = 640
    return rmin, rmax, cmin, cmax


def preprocess_ycb_instance(
    rgb: np.ndarray,
    depth: np.ndarray,
    instance_mask: np.ndarray,
    obj_index: int,
    num_points: int,
    intr: CameraIntrinsics,
    depth_to_meter: float,
    input_h: int,
    input_w: int,
) -> Optional[Dict[str, np.ndarray]]:
    bbox = get_bbox_from_mask(instance_mask)
    if bbox is None:
        return None
    rmin, rmax, cmin, cmax = bbox
    rgb_crop = rgb[rmin:rmax, cmin:cmax, :3]
    depth_crop = depth[rmin:rmax, cmin:cmax]
    mask_crop = (instance_mask[rmin:rmax, cmin:cmax] > 0).astype(np.uint8)
    if rgb_crop.size == 0 or depth_crop.size == 0:
        return None

    rgb_resized = cv2.resize(rgb_crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth_crop, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
    mask_resized = cv2.resize(mask_crop, (input_w, input_h), interpolation=cv2.INTER_NEAREST)

    choose = ((mask_resized != 0) & (depth_resized > 0)).flatten().nonzero()[0]
    if choose.size == 0:
        return None
    if choose.size > num_points:
        c_mask = np.zeros(choose.size, dtype=np.int32)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, num_points - choose.size), mode="wrap")

    row_coords = np.linspace(rmin, rmax - 1, input_h, dtype=np.float32) if input_h > 1 else np.array([float(rmin)], dtype=np.float32)
    col_coords = np.linspace(cmin, cmax - 1, input_w, dtype=np.float32) if input_w > 1 else np.array([float(cmin)], dtype=np.float32)
    xmap = np.tile(row_coords.reshape(-1, 1), (1, input_w))
    ymap = np.tile(col_coords.reshape(1, -1), (input_h, 1))

    depth_masked = depth_resized.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
    pt2 = depth_masked * float(depth_to_meter)
    pt0 = (ymap_masked - intr.cam_cx) * pt2 / intr.cam_fx
    pt1 = (xmap_masked - intr.cam_cy) * pt2 / intr.cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1).astype(np.float32)

    choose = choose.reshape(1, 1, -1).astype(np.int64)
    rgb_chw = np.transpose(rgb_resized.astype(np.float32), (2, 0, 1)) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb_norm = ((rgb_chw - mean) / std).astype(np.float32)

    return {
        "points": cloud[np.newaxis, :, :],
        "choose": choose,
        "img": rgb_norm[np.newaxis, :, :, :],
        "idx": np.array([obj_index], dtype=np.int64),
    }


def infer_pose_onnx_timed(
    pose_sess: ort.InferenceSession,
    refine_sess: ort.InferenceSession,
    data: Dict[str, np.ndarray],
    iteration: int,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    t0 = time.perf_counter()
    pred_r, pred_t, pred_c, emb = pose_sess.run(
        None,
        {"img": data["img"], "points": data["points"], "choose": data["choose"], "obj": data["idx"]},
    )
    t1 = time.perf_counter()
    pred_r_norm = np.linalg.norm(pred_r, axis=2, keepdims=True)
    pred_r_norm[pred_r_norm < 1e-8] = 1.0
    pred_r = pred_r / pred_r_norm
    pred_c = pred_c.reshape(1, num_points)
    which_max = int(np.argmax(pred_c, axis=1)[0])
    pred_t = pred_t.reshape(num_points, 1, 3)
    my_r = pred_r[0][which_max].reshape(-1)
    my_t = (data["points"].reshape(num_points, 1, 3) + pred_t)[which_max].reshape(-1)

    t2 = time.perf_counter()
    for _ in range(iteration):
        t_tensor = np.repeat(my_t.astype(np.float32).reshape(1, 1, 3), num_points, axis=1)
        mat = quaternion_matrix(my_r)
        r_tensor = mat[:3, :3].astype(np.float32).reshape(1, 3, 3)
        mat[0:3, 3] = my_t
        new_points = np.matmul((data["points"] - t_tensor), r_tensor)
        pred_r_refine, pred_t_refine = refine_sess.run(
            None,
            {"points": new_points.astype(np.float32), "emb": emb.astype(np.float32), "obj": data["idx"]},
        )
        pred_r_refine = pred_r_refine.reshape(1, 1, -1)
        refine_norm = np.linalg.norm(pred_r_refine, axis=2, keepdims=True)
        refine_norm[refine_norm < 1e-8] = 1.0
        pred_r_refine = pred_r_refine / refine_norm
        my_r_2 = pred_r_refine.reshape(-1)
        my_t_2 = pred_t_refine.reshape(-1)
        mat_2 = quaternion_matrix(my_r_2)
        mat_2[0:3, 3] = my_t_2
        mat_final = np.dot(mat, mat_2)
        r_final = mat_final.copy()
        r_final[0:3, 3] = 0
        my_r = quaternion_from_matrix(r_final, False)
        q_norm = np.linalg.norm(my_r)
        if q_norm > 1e-8:
            my_r = my_r / q_norm
        my_t = np.array([mat_final[0][3], mat_final[1][3], mat_final[2][3]], dtype=np.float32)
    t3 = time.perf_counter()

    timings = {"pose_ms": (t1 - t0) * 1000.0, "refine_ms": (t3 - t2) * 1000.0, "total_ms": (t3 - t0) * 1000.0}
    return my_r.astype(np.float32), my_t.astype(np.float32), float(pred_c[0, which_max]), timings


@torch.no_grad()
def infer_pose_torch_timed(
    estimator: PoseNet,
    refiner: PoseRefineNet,
    data: Dict[str, np.ndarray],
    iteration: int,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    img = torch.from_numpy(data["img"]).to(torch.float32)
    points = torch.from_numpy(data["points"]).to(torch.float32)
    choose = torch.from_numpy(data["choose"]).to(torch.long)
    idx = torch.from_numpy(data["idx"]).to(torch.long)

    t0 = time.perf_counter()
    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    t1 = time.perf_counter()
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(1, num_points)
    max_conf, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().numpy()
    my_t = (points.view(num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().numpy()

    t2 = time.perf_counter()
    for _ in range(iteration):
        t_tensor = torch.from_numpy(my_t.astype(np.float32)).view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        mat = quaternion_matrix(my_r)
        r_tensor = torch.from_numpy(mat[:3, :3].astype(np.float32)).view(1, 3, 3)
        mat[0:3, 3] = my_t
        new_points = torch.bmm((points - t_tensor), r_tensor).contiguous()
        pred_r_refine, pred_t_refine = refiner(new_points, emb, idx)
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
        my_t = np.array([mat_final[0][3], mat_final[1][3], mat_final[2][3]], dtype=np.float32)
    t3 = time.perf_counter()

    timings = {"pose_ms": (t1 - t0) * 1000.0, "refine_ms": (t3 - t2) * 1000.0, "total_ms": (t3 - t0) * 1000.0}
    return my_r.astype(np.float32), my_t.astype(np.float32), float(max_conf.item()), timings


def quaternion_l2_with_sign(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)))


def main() -> None:
    args = parse_args()
    if args.obj_id < 0 or args.obj_id > 21:
        raise ValueError("--obj_id must be 0 or in [1, 21]")

    torch.manual_seed(0)
    np.random.seed(0)

    estimator, refiner = build_torch_models(args.torch_model, args.torch_refine_model, args.num_points, num_obj=21)
    pose_sess = ort.InferenceSession(args.onnx_pose_model, providers=["CPUExecutionProvider"])
    refine_sess = ort.InferenceSession(args.onnx_refine_model, providers=["CPUExecutionProvider"])

    use_bop_scene = bool(args.bop_scene_dir)
    if not use_bop_scene and not args.dataset_root:
        raise ValueError("Either --bop_scene_dir or --dataset_root must be provided.")
    if use_bop_scene:
        scene_dir = Path(args.bop_scene_dir)
        scene_camera, scene_gt, frame_ids = load_bop_scene_data(scene_dir)
        if args.num_samples > 0:
            frame_ids = frame_ids[: args.num_samples]
    else:
        frame_ids = load_test_ids(args.test_list, args.num_samples)

    preprocess_ms: List[float] = []
    torch_total_ms: List[float] = []
    torch_pose_ms: List[float] = []
    torch_refine_ms: List[float] = []
    onnx_total_ms: List[float] = []
    onnx_pose_ms: List[float] = []
    onnx_refine_ms: List[float] = []
    q_l2_list: List[float] = []
    t_l2_mm_list: List[float] = []
    conf_abs_list: List[float] = []

    per_instance_records: List[Dict[str, object]] = []
    warmup_left = int(args.warmup)
    processed_instances = 0

    for sid in frame_ids:
        if use_bop_scene:
            frame_id = int(sid)
            color_path = scene_dir / "rgb" / f"{frame_id:06d}.jpg"
            depth_path = scene_dir / "depth" / f"{frame_id:06d}.png"
            if not (color_path.exists() and depth_path.exists()):
                continue
            rgb_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or depth is None:
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            cam_k = scene_camera[sid]["cam_K"]
            intr = CameraIntrinsics(cam_fx=float(cam_k[0]), cam_fy=float(cam_k[4]), cam_cx=float(cam_k[2]), cam_cy=float(cam_k[5]))
            depth_scale = float(scene_camera[sid].get("depth_scale", 0.1))
            # BOP convention: depth_mm = depth_raw * depth_scale, then convert to meters.
            depth_to_meter = depth_scale / 1000.0
            gt_list = scene_gt[sid]
            instance_iter = []
            for inst_id, gt in enumerate(gt_list):
                itemid = int(gt["obj_id"])
                if args.obj_id != 0 and itemid != args.obj_id:
                    continue
                mask_path = scene_dir / "mask_visib" / f"{frame_id:06d}_{inst_id:06d}.png"
                if not mask_path.exists():
                    continue
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue
                instance_mask = (mask > 0).astype(np.uint8)
                instance_iter.append((itemid, instance_mask, depth_to_meter))
        else:
            color_path = Path(args.dataset_root) / f"{sid}-color.png"
            depth_path = Path(args.dataset_root) / f"{sid}-depth.png"
            label_path = Path(args.dataset_root) / f"{sid}-label.png"
            meta_path = Path(args.dataset_root) / f"{sid}-meta.mat"
            if not (color_path.exists() and depth_path.exists() and label_path.exists() and meta_path.exists()):
                continue
            rgb = cv2.cvtColor(cv2.imread(str(color_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None or label is None:
                continue
            meta = scio.loadmat(str(meta_path))
            if "cls_indexes" not in meta:
                continue
            cls_indexes = meta["cls_indexes"].flatten().astype(np.int32).tolist()
            factor_depth = float(meta.get("factor_depth", np.array([[10000.0]])).reshape(-1)[0])
            depth_to_meter = 1.0 / factor_depth
            intr = get_intrinsics_for_sid(sid)
            instance_iter = []
            for itemid in cls_indexes:
                if args.obj_id != 0 and itemid != args.obj_id:
                    continue
                instance_iter.append((int(itemid), (label == itemid).astype(np.uint8), depth_to_meter))

        for itemid, instance_mask, depth_to_meter in instance_iter:
            obj_index = int(itemid - 1)

            t_pre0 = time.perf_counter()
            data = preprocess_ycb_instance(
                rgb=rgb,
                depth=depth.astype(np.float32),
                instance_mask=instance_mask,
                obj_index=obj_index,
                num_points=args.num_points,
                intr=intr,
                depth_to_meter=depth_to_meter,
                input_h=args.input_h,
                input_w=args.input_w,
            )
            t_pre1 = time.perf_counter()
            if data is None:
                continue

            # Warmup: run both backends but skip metric collection.
            if warmup_left > 0:
                _ = infer_pose_torch_timed(estimator, refiner, data, args.iteration, args.num_points)
                _ = infer_pose_onnx_timed(pose_sess, refine_sess, data, args.iteration, args.num_points)
                warmup_left -= 1
                continue

            preprocess_ms.append((t_pre1 - t_pre0) * 1000.0)
            tq, tt, tc, t_times = infer_pose_torch_timed(estimator, refiner, data, args.iteration, args.num_points)
            oq, ot, oc, o_times = infer_pose_onnx_timed(pose_sess, refine_sess, data, args.iteration, args.num_points)

            torch_pose_ms.append(t_times["pose_ms"])
            torch_refine_ms.append(t_times["refine_ms"])
            torch_total_ms.append(t_times["total_ms"])
            onnx_pose_ms.append(o_times["pose_ms"])
            onnx_refine_ms.append(o_times["refine_ms"])
            onnx_total_ms.append(o_times["total_ms"])

            q_l2 = quaternion_l2_with_sign(tq, oq)
            t_l2_mm = float(np.linalg.norm(tt - ot) * 1000.0)
            conf_abs = float(abs(tc - oc))
            q_l2_list.append(q_l2)
            t_l2_mm_list.append(t_l2_mm)
            conf_abs_list.append(conf_abs)
            processed_instances += 1

            per_instance_records.append(
                {
                    "sid": sid,
                    "obj_id": int(itemid),
                    "preprocess_ms": preprocess_ms[-1],
                    "torch_timing_ms": t_times,
                    "onnx_timing_ms": o_times,
                    "pose_diff": {
                        "quat_l2_sign_aware": q_l2,
                        "trans_l2_mm": t_l2_mm,
                        "conf_abs": conf_abs,
                    },
                }
            )

    pre_stat = summarize(preprocess_ms)
    torch_pose_stat = summarize(torch_pose_ms)
    torch_refine_stat = summarize(torch_refine_ms)
    torch_total_stat = summarize(torch_total_ms)
    onnx_pose_stat = summarize(onnx_pose_ms)
    onnx_refine_stat = summarize(onnx_refine_ms)
    onnx_total_stat = summarize(onnx_total_ms)
    q_stat = summarize(q_l2_list)
    t_stat = summarize(t_l2_mm_list)
    c_stat = summarize(conf_abs_list)

    print("=== DenseFusion YCB Benchmark: PyTorch vs ONNX (CPU) ===")
    print(f"frames_scanned={len(frame_ids)} processed_instances={processed_instances} warmup_instances={args.warmup}")
    print(
        f"settings: obj_id={args.obj_id} num_points={args.num_points} iteration={args.iteration} input={args.input_h}x{args.input_w}"
    )
    print(f"preprocess ms     : avg={pre_stat['avg']:.3f} p50={pre_stat['p50']:.3f} p90={pre_stat['p90']:.3f}")
    print(
        f"pytorch pose ms   : avg={torch_pose_stat['avg']:.3f} p50={torch_pose_stat['p50']:.3f} p90={torch_pose_stat['p90']:.3f}"
    )
    print(
        f"pytorch refine ms : avg={torch_refine_stat['avg']:.3f} p50={torch_refine_stat['p50']:.3f} p90={torch_refine_stat['p90']:.3f}"
    )
    print(
        f"pytorch total ms  : avg={torch_total_stat['avg']:.3f} p50={torch_total_stat['p50']:.3f} p90={torch_total_stat['p90']:.3f}"
    )
    print(
        f"onnx pose ms      : avg={onnx_pose_stat['avg']:.3f} p50={onnx_pose_stat['p50']:.3f} p90={onnx_pose_stat['p90']:.3f}"
    )
    print(
        f"onnx refine ms    : avg={onnx_refine_stat['avg']:.3f} p50={onnx_refine_stat['p50']:.3f} p90={onnx_refine_stat['p90']:.3f}"
    )
    print(
        f"onnx total ms     : avg={onnx_total_stat['avg']:.3f} p50={onnx_total_stat['p50']:.3f} p90={onnx_total_stat['p90']:.3f}"
    )
    if onnx_total_stat["avg"] > 0:
        print(f"speedup torch/onnx (total): {torch_total_stat['avg'] / onnx_total_stat['avg']:.3f}x")
    print(f"quat l2(sign-aware): avg={q_stat['avg']:.6e} p50={q_stat['p50']:.6e} p90={q_stat['p90']:.6e}")
    print(f"trans l2 diff (mm): avg={t_stat['avg']:.6f} p50={t_stat['p50']:.6f} p90={t_stat['p90']:.6f}")
    print(f"confidence abs diff: avg={c_stat['avg']:.6e} p50={c_stat['p50']:.6e} p90={c_stat['p90']:.6e}")

    if args.output_json:
        output = {
            "summary": {
                "frames_scanned": len(frame_ids),
                "processed_instances": processed_instances,
                "warmup_instances": args.warmup,
                "preprocess_ms": pre_stat,
                "pytorch_pose_ms": torch_pose_stat,
                "pytorch_refine_ms": torch_refine_stat,
                "pytorch_total_ms": torch_total_stat,
                "onnx_pose_ms": onnx_pose_stat,
                "onnx_refine_ms": onnx_refine_stat,
                "onnx_total_ms": onnx_total_stat,
                "pose_diff_quat_l2_sign_aware": q_stat,
                "pose_diff_trans_l2_mm": t_stat,
                "pose_diff_conf_abs": c_stat,
            },
            "records": per_instance_records,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"saved json: {args.output_json}")


if __name__ == "__main__":
    main()
