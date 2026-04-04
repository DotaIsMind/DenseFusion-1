#!/usr/bin/env python3
import argparse
import logging

import torch
import torch.nn as nn

from lib.network import PoseNet, PoseRefineNet


LOGGER = logging.getLogger("densefusion_export_onnx")


def unwrap_dataparallel(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.DataParallel):
        return module.module
    for name, child in module.named_children():
        setattr(module, name, unwrap_dataparallel(child))
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DenseFusion models to ONNX")
    parser.add_argument("--model", type=str, required=True, help="PoseNet checkpoint path")
    parser.add_argument("--refine_model", type=str, default="", help="PoseRefineNet checkpoint path")
    parser.add_argument("--output_pose_onnx", type=str, default="densefusion_posenet.onnx")
    parser.add_argument("--output_refine_onnx", type=str, default="densefusion_refiner.onnx")
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--num_obj", type=int, default=13)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--input_h", type=int, default=80, help="export-time fixed input image height")
    parser.add_argument("--input_w", type=int, default=80, help="export-time fixed input image width")
    return parser.parse_args()


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def export_posenet(args: argparse.Namespace) -> None:
    model = PoseNet(num_points=args.num_points, num_obj=args.num_obj)
    model = model.to("cpu")
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model = unwrap_dataparallel(model)
    model = model.to("cpu")
    model.eval()

    dummy_img = torch.randn(1, 3, args.input_h, args.input_w, dtype=torch.float32)
    dummy_points = torch.randn(1, args.num_points, 3, dtype=torch.float32)
    dummy_choose = torch.randint(0, 120 * 160, (1, 1, args.num_points), dtype=torch.long)
    dummy_obj = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_img, dummy_points, dummy_choose, dummy_obj),
        args.output_pose_onnx,
        dynamo=True,
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["img", "points", "choose", "obj"],
        output_names=["pred_r", "pred_t", "pred_c", "emb"],
    )
    LOGGER.info("Exported PoseNet ONNX: %s", args.output_pose_onnx)


def export_refiner(args: argparse.Namespace) -> None:
    if not args.refine_model:
        LOGGER.info("Skip PoseRefineNet export, no --refine_model provided.")
        return
    model = PoseRefineNet(num_points=args.num_points, num_obj=args.num_obj)
    model = model.to("cpu")
    model.load_state_dict(torch.load(args.refine_model, map_location="cpu"))
    model = unwrap_dataparallel(model)
    model = model.to("cpu")
    model.eval()

    dummy_points = torch.randn(1, args.num_points, 3, dtype=torch.float32)
    dummy_emb = torch.randn(1, 32, args.num_points, dtype=torch.float32)
    dummy_obj = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_points, dummy_emb, dummy_obj),
        args.output_refine_onnx,
        dynamo=True,
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["points", "emb", "obj"],
        output_names=["pred_r", "pred_t"],
    )
    LOGGER.info("Exported PoseRefineNet ONNX: %s", args.output_refine_onnx)


def main() -> None:
    setup_logger()
    args = parse_args()
    export_posenet(args)
    export_refiner(args)


if __name__ == "__main__":
    main()
