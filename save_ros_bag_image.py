#!/usr/bin/env python3
import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class LinemodBagSaver(Node):
    def __init__(self):
        super().__init__("linemod_bag_saver")
        self.declare_parameter("output_root", "./Linemod_preprocessed")
        self.declare_parameter("obj_id", 1)
        self.declare_parameter("split_file", "test.txt")
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("sync_slop", 0.05)

        self.declare_parameter("color_info_topic", "/camera/color/camera_info")
        self.declare_parameter("color_image_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("depth_image_topic", "/camera/depth/image_raw")

        output_root = Path(str(self.get_parameter("output_root").value)).resolve()
        obj_id = int(self.get_parameter("obj_id").value)
        split_file = str(self.get_parameter("split_file").value)

        self.obj_dir = output_root / "data" / f"{obj_id:02d}"
        self.rgb_dir = self.obj_dir / "rgb"
        self.depth_dir = self.obj_dir / "depth"
        self.mask_dir = self.obj_dir / "mask"
        self.meta_dir = self.obj_dir / "meta"
        for d in (self.rgb_dir, self.depth_dir, self.mask_dir, self.meta_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.split_path = self.obj_dir / split_file
        self.seq = self._infer_next_index()

        color_info_topic = str(self.get_parameter("color_info_topic").value)
        color_image_topic = str(self.get_parameter("color_image_topic").value)
        depth_info_topic = str(self.get_parameter("depth_info_topic").value)
        depth_image_topic = str(self.get_parameter("depth_image_topic").value)
        queue_size = int(self.get_parameter("queue_size").value)
        sync_slop = float(self.get_parameter("sync_slop").value)

        self.color_info_sub = Subscriber(self, CameraInfo, color_info_topic)
        self.color_image_sub = Subscriber(self, Image, color_image_topic)
        self.depth_info_sub = Subscriber(self, CameraInfo, depth_info_topic)
        self.depth_image_sub = Subscriber(self, Image, depth_image_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.color_info_sub, self.color_image_sub, self.depth_info_sub, self.depth_image_sub],
            queue_size=queue_size,
            slop=sync_slop,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(f"Saving LINEMOD data to: {self.obj_dir}")
        self.get_logger().info(
            f"Subscribed topics: {color_info_topic}, {color_image_topic}, {depth_info_topic}, {depth_image_topic}"
        )
        self.get_logger().info(f"Next frame index: {self.seq:06d}")

    def _infer_next_index(self) -> int:
        existing = sorted(self.rgb_dir.glob("*.png"))
        if not existing:
            return 0
        last_stem = existing[-1].stem
        try:
            return int(last_stem) + 1
        except ValueError:
            return len(existing)

    def _decode_color(self, msg: Image) -> np.ndarray:
        if msg.encoding not in ("bgr8", "rgb8"):
            raise ValueError(f"Unsupported color encoding: {msg.encoding}")
        channels = 3
        expected_step = msg.width * channels
        if msg.step < expected_step:
            raise ValueError(f"Invalid color step: {msg.step} < {expected_step}")
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.step))
        arr = arr[:, :expected_step].reshape((msg.height, msg.width, channels))
        if msg.encoding == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def _decode_depth_to_mm_u16(self, msg: Image) -> np.ndarray:
        if msg.encoding == "16UC1":
            expected_step = msg.width * 2
            if msg.step < expected_step:
                raise ValueError(f"Invalid depth step for 16UC1: {msg.step} < {expected_step}")
            arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.step // 2))
            return arr[:, : msg.width]
        if msg.encoding == "32FC1":
            expected_step = msg.width * 4
            if msg.step < expected_step:
                raise ValueError(f"Invalid depth step for 32FC1: {msg.step} < {expected_step}")
            arr_m = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.step // 4))
            arr_m = arr_m[:, : msg.width]
            arr_mm = np.where(np.isfinite(arr_m) & (arr_m > 0.0), arr_m * 1000.0, 0.0)
            return np.clip(arr_mm, 0.0, 65535.0).astype(np.uint16)
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

    @staticmethod
    def _camera_info_to_dict(msg: CameraInfo) -> dict:
        return {
            "header": {
                "stamp_sec": int(msg.header.stamp.sec),
                "stamp_nanosec": int(msg.header.stamp.nanosec),
                "frame_id": msg.header.frame_id,
            },
            "height": int(msg.height),
            "width": int(msg.width),
            "distortion_model": msg.distortion_model,
            "d": [float(x) for x in msg.d],
            "k": [float(x) for x in msg.k],
            "r": [float(x) for x in msg.r],
            "p": [float(x) for x in msg.p],
            "binning_x": int(msg.binning_x),
            "binning_y": int(msg.binning_y),
        }

    def synced_callback(
        self,
        color_info_msg: CameraInfo,
        color_img_msg: Image,
        depth_info_msg: CameraInfo,
        depth_img_msg: Image,
    ) -> None:
        try:
            color_bgr = self._decode_color(color_img_msg)
            depth_mm_u16 = self._decode_depth_to_mm_u16(depth_img_msg)
            mask = np.where(depth_mm_u16 > 0, 255, 0).astype(np.uint8)

            sid = f"{self.seq:06d}"
            rgb_path = self.rgb_dir / f"{sid}.png"
            depth_path = self.depth_dir / f"{sid}.png"
            mask_path = self.mask_dir / f"{sid}.png"
            meta_path = self.meta_dir / f"{sid}.json"

            cv2.imwrite(str(rgb_path), color_bgr)
            cv2.imwrite(str(depth_path), depth_mm_u16)
            cv2.imwrite(str(mask_path), mask)

            meta = {
                "frame_id": sid,
                "color_info": self._camera_info_to_dict(color_info_msg),
                "depth_info": self._camera_info_to_dict(depth_info_msg),
                "color_image_encoding": color_img_msg.encoding,
                "depth_image_encoding": depth_img_msg.encoding,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            with open(self.split_path, "a", encoding="utf-8") as f:
                f.write(f"{sid}\n")

            self.get_logger().info(
                f"Saved frame {sid} | rgb={rgb_path.name} depth={depth_path.name} mask={mask_path.name}"
            )
            self.seq += 1
        except Exception as exc:
            self.get_logger().error(f"Failed to save synced frame: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = LinemodBagSaver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()