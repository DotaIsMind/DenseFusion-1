import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

_RGB_EXTS = (".jpg", ".jpeg", ".png")
_DEPTH_EXTS = (".png", ".tiff", ".tif", ".jpg")


class FileInputPublisher(Node):
    """发布 RGB + Depth 图像到 ROS topic。

    支持两种模式：
    - 目录模式（推荐）：设置 ``scene_dir``，自动扫描 ``<scene_dir>/rgb/``
      和 ``<scene_dir>/depth/`` 下所有匹配帧，按文件名顺序逐帧发布。
    - 单图模式（兼容旧配置）：同时设置 ``rgb_path`` + ``depth_path``，
      反复发布同一帧。
    """

    def __init__(self):
        super().__init__("file_input_publisher")

        # ── 参数声明 ──────────────────────────────────────────────────────────
        self.declare_parameter("scene_dir", "")       # 目录模式：scene 根目录
        self.declare_parameter("rgb_path", "")        # 单图模式：RGB 文件路径
        self.declare_parameter("depth_path", "")      # 单图模式：Depth 文件路径
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("publish_hz", 5.0)
        self.declare_parameter("loop", False)         # 目录模式：播放完后是否循环

        scene_dir   = self.get_parameter("scene_dir").value.strip()
        rgb_path    = self.get_parameter("rgb_path").value.strip()
        depth_path  = self.get_parameter("depth_path").value.strip()
        self.rgb_topic    = self.get_parameter("rgb_topic").value
        self.depth_topic  = self.get_parameter("depth_topic").value
        self.publish_hz   = float(self.get_parameter("publish_hz").value)
        self.loop         = bool(self.get_parameter("loop").value)

        # ── 构建帧列表 ────────────────────────────────────────────────────────
        self.frames: List[Tuple[str, str]] = []  # [(rgb_path, depth_path), ...]

        if scene_dir:
            self._load_scene_dir(scene_dir)
        elif rgb_path and depth_path:
            self._load_single(rgb_path, depth_path)
        else:
            raise ValueError(
                "必须提供 'scene_dir'，或同时提供 'rgb_path' 与 'depth_path'。"
            )

        if not self.frames:
            raise RuntimeError("未找到任何有效的 RGB/Depth 帧对，请检查路径。")

        self.frame_idx = 0

        self.rgb_pub   = self.create_publisher(Image, self.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.timer = self.create_timer(max(0.01, 1.0 / self.publish_hz), self._tick)

        self.get_logger().info(
            f"FileInputPublisher: {len(self.frames)} 帧，"
            f"发布至 {self.rgb_topic} / {self.depth_topic}，"
            f"{self.publish_hz:.1f} Hz，loop={self.loop}"
        )

    # ── 帧列表构建 ────────────────────────────────────────────────────────────

    def _load_single(self, rgb_path: str, depth_path: str) -> None:
        """单图模式：校验文件后加入帧列表，tick 时反复发布。"""
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB 文件不存在: {rgb_path}")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth 文件不存在: {depth_path}")
        # 单图模式也设置 loop=True，保持原来反复发布的行为
        self.loop = True
        self.frames.append((rgb_path, depth_path))
        self.get_logger().info(f"单图模式：{os.path.basename(rgb_path)}")

    def _load_scene_dir(self, scene_dir: str) -> None:
        """目录模式：扫描 <scene_dir>/rgb/ 与 <scene_dir>/depth/，匹配同名帧。"""
        rgb_dir   = os.path.join(scene_dir, "rgb")
        depth_dir = os.path.join(scene_dir, "depth")

        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"RGB 子目录不存在: {rgb_dir}")
        if not os.path.isdir(depth_dir):
            raise FileNotFoundError(f"Depth 子目录不存在: {depth_dir}")

        # 收集所有 RGB 文件，按文件名排序
        rgb_files: List[str] = []
        for ext in _RGB_EXTS:
            rgb_files.extend(glob.glob(os.path.join(rgb_dir, f"*{ext}")))
        rgb_files = sorted(set(rgb_files), key=os.path.basename)

        skipped = 0
        for rgb_file in rgb_files:
            stem = os.path.splitext(os.path.basename(rgb_file))[0]
            depth_file = self._find_depth(depth_dir, stem)
            if depth_file is None:
                self.get_logger().warn(
                    f"RGB {os.path.basename(rgb_file)} 无对应 depth 文件，已跳过。"
                )
                skipped += 1
                continue
            self.frames.append((rgb_file, depth_file))

        self.get_logger().info(
            f"目录模式：scene_dir={scene_dir}，"
            f"共 {len(self.frames)} 帧，跳过 {skipped} 帧。"
        )

    @staticmethod
    def _find_depth(depth_dir: str, stem: str) -> str | None:
        """按 stem 在 depth_dir 中查找匹配的深度图文件。"""
        for ext in _DEPTH_EXTS:
            candidate = os.path.join(depth_dir, f"{stem}{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    # ── 图像工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_image_msg(arr: np.ndarray, encoding: str, stamp, frame_id: str) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(arr.shape[0])
        msg.width  = int(arr.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = int(arr.strides[0])
        msg.data = np.ascontiguousarray(arr).tobytes()
        return msg

    @staticmethod
    def _ensure_depth_u16(depth: np.ndarray) -> np.ndarray:
        if depth.dtype == np.uint16:
            return depth
        if depth.dtype in (np.float32, np.float64):
            depth_mm = depth.copy()
            if float(depth_mm.max()) < 100.0:   # 判断是否为米单位
                depth_mm = depth_mm * 1000.0
            return depth_mm.clip(0.0, 65535.0).astype(np.uint16)
        return depth.astype(np.uint16)

    # ── 定时回调 ──────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self.frame_idx >= len(self.frames):
            if self.loop:
                self.frame_idx = 0
                self.get_logger().info("所有帧已播放完毕，循环回到第 1 帧。")
            else:
                self.get_logger().info("所有帧已播放完毕，停止发布。")
                self.timer.cancel()
                return

        rgb_path, depth_path = self.frames[self.frame_idx]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb is None or depth is None:
            self.get_logger().warn(
                f"第 {self.frame_idx + 1} 帧加载失败，已跳过：{rgb_path}"
            )
            self.frame_idx += 1
            return

        depth = self._ensure_depth_u16(depth)

        if depth.shape[:2] != rgb.shape[:2]:
            self.get_logger().warn(
                f"第 {self.frame_idx + 1} 帧分辨率不匹配 "
                f"rgb={rgb.shape[:2]} depth={depth.shape[:2]}，已跳过。"
            )
            self.frame_idx += 1
            return

        stamp = self.get_clock().now().to_msg()
        self.rgb_pub.publish(
            self._to_image_msg(rgb, "bgr8", stamp, "camera_color_frame")
        )
        self.depth_pub.publish(
            self._to_image_msg(depth, "16UC1", stamp, "camera_depth_frame")
        )

        self.get_logger().info(
            f"[{self.frame_idx + 1}/{len(self.frames)}] {os.path.basename(rgb_path)}"
        )
        self.frame_idx += 1


def main():
    rclpy.init()
    node = FileInputPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
