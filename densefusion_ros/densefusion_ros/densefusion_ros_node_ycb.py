import os
import sys
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray


def _load_pose_result_msg_type():
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    probe_roots = [Path.cwd(), Path(__file__).resolve()]
    probe_roots.extend(Path(__file__).resolve().parents)
    generated_sites = []
    for root in probe_roots:
        candidates = [
            root / "install" / "densefusion_ros" / "lib" / pyver / "site-packages",
            root / "densefusion_ros" / "install" / "densefusion_ros" / "lib" / pyver / "site-packages",
        ]
        for cand in candidates:
            if (cand / "densefusion_ros" / "msg").is_dir():
                generated_sites.append(str(cand))

    for path in list(sys.path):
        p = Path(path)
        if (p / "densefusion_ros" / "msg").is_dir():
            generated_sites.append(str(p))

    for site in reversed(list(dict.fromkeys(generated_sites))):
        if site in sys.path:
            sys.path.remove(site)
        sys.path.insert(0, site)

    loaded_pkg = sys.modules.get("densefusion_ros")
    loaded_file = str(getattr(loaded_pkg, "__file__", ""))
    if loaded_pkg is not None and "/site-packages/" not in loaded_file:
        sys.modules.pop("densefusion_ros", None)
        sys.modules.pop("densefusion_ros.msg", None)

    from densefusion_ros.msg import PoseEstimationResult

    PoseEstimationResult.__class__.__import_type_support__()
    if PoseEstimationResult.__class__._TYPE_SUPPORT is None:
        raise RuntimeError(
            "PoseEstimationResult type support is unavailable. "
            "Please rebuild and source this workspace."
        )
    return PoseEstimationResult


PoseEstimationResult = _load_pose_result_msg_type()

from .densefusion_core import CameraIntrinsics, OnnxDenseFusion, Yolo11SegOnnx, preprocess_rgbd, quaternion_matrix
from .densefusion_core.preprocess import infer_pose_onnx


YCB_OBJ_IDS = tuple(range(1, 22))


class DenseFusionRosNodeYcb(Node):
    def __init__(self):
        super().__init__("densefusion_ros_node_ycb")
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("mask_topic", "")
        self.declare_parameter("obj_id", 11)  # banana
        self.declare_parameter("num_points", 1000)
        self.declare_parameter("iteration", 2)
        self.declare_parameter(
            "pose_onnx_path",
            "/home/data/qrb_ros_simulation_ws/DenseFusion-1/ycb-data-onnx-model/densefusion_ycb_posenet.onnx",
        )
        self.declare_parameter(
            "refine_onnx_path",
            "/home/data/qrb_ros_simulation_ws/DenseFusion-1/ycb-data-onnx-model/densefusion_ycb_refiner.onnx",
        )
        self.declare_parameter(
            "yolo_seg_onnx_path",
            "/home/data/qrb_ros_simulation_ws/DenseFusion-1/yolo11-seg-model/yolo26n-seg.onnx",
        )
        self.declare_parameter("target_label", "banana")
        self.declare_parameter("target_conf", 0.25)
        self.declare_parameter("yolo_score_th", 0.25)
        self.declare_parameter("yolo_mask_th", 0.5)
        # self.declare_parameter("cam_fx", 1066.778)
        # self.declare_parameter("cam_fy", 1067.487)
        # self.declare_parameter("cam_cx", 312.9869)
        # self.declare_parameter("cam_cy", 241.3109)
        self.declare_parameter("cam_fx", 461.07720947265625)
        self.declare_parameter("cam_fy", 461.29638671875)
        self.declare_parameter("cam_cx", 318.0372009277344)
        self.declare_parameter("cam_cy", 236.3270721435547)
        self.declare_parameter("depth_scale", 1.0)
        self.declare_parameter("input_h", 80)
        self.declare_parameter("input_w", 80)
        self.declare_parameter("save_vis", True)
        self.declare_parameter("save_video", False)
        self.declare_parameter("video_fps", 15.0)
        self.declare_parameter("vis_output_dir", "benchmark-ouputs/ros-ycb-node")
        self.declare_parameter("save_vis_every_n", 1)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.mask_topic = str(self.get_parameter("mask_topic").value).strip()
        self.obj_id = int(self.get_parameter("obj_id").value)
        self.num_points = int(self.get_parameter("num_points").value)
        self.iteration = int(self.get_parameter("iteration").value)
        self.pose_onnx_path = str(self.get_parameter("pose_onnx_path").value)
        self.refine_onnx_path = str(self.get_parameter("refine_onnx_path").value)
        self.yolo_seg_onnx_path = str(self.get_parameter("yolo_seg_onnx_path").value)
        self.target_label = str(self.get_parameter("target_label").value).strip().lower()
        self.target_conf = float(self.get_parameter("target_conf").value)
        self.yolo_score_th = float(self.get_parameter("yolo_score_th").value)
        self.yolo_mask_th = float(self.get_parameter("yolo_mask_th").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.input_h = int(self.get_parameter("input_h").value)
        self.input_w = int(self.get_parameter("input_w").value)
        self.save_vis = bool(self.get_parameter("save_vis").value)
        self.save_video = bool(self.get_parameter("save_video").value)
        self.video_fps = float(self.get_parameter("video_fps").value)
        self.vis_output_dir = str(self.get_parameter("vis_output_dir").value + f"/{self.target_label}")
        self.save_vis_every_n = int(self.get_parameter("save_vis_every_n").value)

        if self.obj_id not in YCB_OBJ_IDS:
            raise ValueError(f"obj_id must be in {YCB_OBJ_IDS}, got {self.obj_id}")

        self.intr = CameraIntrinsics(
            cam_fx=float(self.get_parameter("cam_fx").value),
            cam_fy=float(self.get_parameter("cam_fy").value),
            cam_cx=float(self.get_parameter("cam_cx").value),
            cam_cy=float(self.get_parameter("cam_cy").value),
        )
        if not os.path.exists(self.pose_onnx_path) or not os.path.exists(self.refine_onnx_path):
            raise FileNotFoundError("pose_onnx_path/refine_onnx_path not found")

        self.obj_index = self.obj_id - 1
        self.runner = OnnxDenseFusion(self.pose_onnx_path, self.refine_onnx_path)
        self.yolo_seg: Optional[Yolo11SegOnnx] = None
        if not self.mask_topic:
            if not os.path.exists(self.yolo_seg_onnx_path):
                raise FileNotFoundError(f"YOLO ONNX model not found: {self.yolo_seg_onnx_path}")
            self.yolo_seg = Yolo11SegOnnx(self.yolo_seg_onnx_path, score_th=self.yolo_score_th, mask_th=self.yolo_mask_th)

        if self.save_vis:
            if not os.path.exists(self.vis_output_dir):
                os.makedirs(self.vis_output_dir, exist_ok=True)
            # else:
            #     import shutil
            #     shutil.remove(self.vis_output_dir)
    
        self.frame_counter = 0
        self.video_writer = None
        self.video_path = os.path.join(self.vis_output_dir, f"{self.target_label}_pose.mp4")

        self.pose_pub = self.create_publisher(PoseStamped, "/pose_stamp", 10)
        self.offset_pub = self.create_publisher(Vector3Stamped, "/pose_stamp_offset", 10)
        self.rot_mat_pub = self.create_publisher(Float64MultiArray, "/pose_stamp_rotation_matrix", 10)
        self.pose_result_pub = self.create_publisher(PoseEstimationResult, "/pose_estimation_result", 10)
        self.pose_result_frame_id = 0

        # Keep only the newest sensor frame to avoid processing stale backlog.
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=sensor_qos)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        subs = [self.rgb_sub, self.depth_sub]
        if self.mask_topic:
            self.mask_sub = Subscriber(self, Image, self.mask_topic, qos_profile=sensor_qos)
            subs.append(self.mask_sub)
        self.sync = ApproximateTimeSynchronizer(subs, queue_size=1, slop=0.03)
        self.sync.registerCallback(self._on_sync)

    def _image_to_numpy(self, msg: Image) -> np.ndarray:
        dtype = {"rgb8": np.uint8, "bgr8": np.uint8, "mono8": np.uint8, "16UC1": np.uint16, "32FC1": np.float32}.get(msg.encoding)
        if dtype is None:
            raise ValueError(f"Unsupported image encoding: {msg.encoding}")
        channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
        row_elems = msg.step // np.dtype(dtype).itemsize
        arr = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, row_elems))
        if channels == 1:
            arr = arr[:, : msg.width]
        else:
            arr = arr[:, : msg.width * channels].reshape((msg.height, msg.width, channels))
        if msg.is_bigendian and np.dtype(dtype).itemsize > 1:
            arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
        return np.ascontiguousarray(arr)

    def _to_rgb(self, rgb_msg: Image) -> np.ndarray:
        img = self._image_to_numpy(rgb_msg)
        if rgb_msg.encoding == "bgr8":
            return img[:, :, ::-1]
        if rgb_msg.encoding == "rgb8":
            return img
        raise ValueError(f"Unsupported RGB encoding: {rgb_msg.encoding}")

    def _to_depth_mm(self, depth_msg: Image) -> np.ndarray:
        depth = self._image_to_numpy(depth_msg).astype(np.float32)
        if depth_msg.encoding == "32FC1":
            # 32FC1 以米为单位，固定乘以 1000 转为毫米
            return depth * 1000.0
        # 16UC1 / mono16：乘以 depth_scale（默认 1.0，即已经是毫米；
        # YCB 原始数据单位为 0.1mm，需在 launch 文件中设置 depth_scale=0.1）
        return depth * self.depth_scale

    def _msg_to_mask_u8(self, mask_msg: Image) -> np.ndarray:
        mask = self._image_to_numpy(mask_msg)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0).astype(np.uint8) * 255

    def _on_sync(self, *msgs):
        rgb_msg, depth_msg = msgs[0], msgs[1]
        mask_msg = msgs[2] if len(msgs) > 2 else None

        rgb = self._to_rgb(rgb_msg)
        depth = self._to_depth_mm(depth_msg)

        other_instances = []
        if mask_msg is not None:
            mask = self._msg_to_mask_u8(mask_msg)
        else:
            if self.yolo_seg is None:
                self.get_logger().warn("No mask source available; frame skipped.")
                return

            all_instances = self.yolo_seg.infer_instances(rgb)
            target_inst = None
            for inst in all_instances:
                if inst["label"] == self.target_label:
                    if inst["score"] < self.target_conf:
                        continue
                    x, y, w, h = cv2.boundingRect(cv2.findNonZero(inst["mask"][:, :, 0] if inst["mask"].ndim == 3 else inst["mask"]))
                    area = w * h
                    image_area = rgb.shape[0] * rgb.shape[1]
                    # if area / image_area < (1/12):
                    #     self.get_logger().warn(f"Object area is less than 1/12 of image area; frame skipped.")
                    #     continue
                    if target_inst is None or inst["score"] > target_inst["score"]:
                        target_inst = inst
                else:
                    other_instances.append(inst)
                    # self.get_logger().warn(
                    #     f"Non-target object detected: '{inst['label']}' "
                    #     f"(class_id={inst['class_id']}, score={inst['score']:.2f}) — ignored for pose estimation."
                    # )

            if target_inst is None:
                # self.get_logger().warn(
                #     f"YOLO '{self.target_label}' mask unavailable with target_conf>={self.target_conf:.2f}; fallback to depth>0 mask."
                # )
                # mask = (depth > 0).astype(np.uint8) * 255
                return
            else:
                mask = target_inst["mask"]

        data = preprocess_rgbd(rgb, depth, mask, self.obj_index, self.num_points, self.intr, input_h=self.input_h, input_w=self.input_w)
        if data is None:
            self.get_logger().warn("No valid points from depth/mask; frame skipped.")
            return

        quat, trans, conf = infer_pose_onnx(self.runner, data, self.iteration, self.num_points)
        rot = quaternion_matrix(quat)[:3, :3]
        if not self._check_pose(quat, trans, conf):
            return
        self._publish_pose(rgb_msg, quat, trans, rot)
        roll, pitch, yaw = self._rot_to_euler_deg(rot)
        rot_str = np.array2string(rot, precision=4, suppress_small=True, separator=", ").replace("\n", "")
        self.get_logger().info(
            f"Published pose | label={self.target_label} conf={conf:.4f} "
            f"R={rot_str} "
            f"euler_rpy_deg=(roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}) "
            f"T_xyz_m=({trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f})"
        )
        self._save_visualization(rgb_msg, rgb, mask, rot, trans, conf, other_instances)

    def _check_pose(self, quat: np.ndarray, trans: np.ndarray, conf: float):
        if not np.all(np.isfinite(quat)) or not np.all(np.isfinite(trans)):
            self.get_logger().warn("Invalid pose; frame skipped.")
            return False
        if trans[2] < 0:
            self.get_logger().warn("Trans z is negative; frame skipped.")
            return False
        if trans[2] > 0.6:
            self.get_logger().warn("Trans z is greater than 0.6; frame skipped.")
            return False
        if conf < 0.2:
            self.get_logger().warn("Confidence is less than 0.2; frame skipped.")
            return False
        return True

    def _project(self, pt3: np.ndarray):
        x, y, z = float(pt3[0]), float(pt3[1]), float(pt3[2])
        if z <= 1e-6 or not np.isfinite([x, y, z]).all():
            return None
        return int(round((x * self.intr.cam_fx) / z + self.intr.cam_cx)), int(round((y * self.intr.cam_fy) / z + self.intr.cam_cy))

    @staticmethod
    def _rot_to_euler_deg(rot: np.ndarray):
        """将 3×3 旋转矩阵转换为 ZYX 欧拉角（度），返回 (roll, pitch, yaw)。

        ZYX 顺序：先绕 Z 轴（yaw），再绕 Y 轴（pitch），最后绕 X 轴（roll）。
        奇异点（pitch ≈ ±90°）时 roll 退化为 0。
        """
        sy = float(np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
        if sy >= 1e-6:
            roll  = np.degrees(np.arctan2( rot[2, 1], rot[2, 2]))
            pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
            yaw   = np.degrees(np.arctan2( rot[1, 0], rot[0, 0]))
        else:
            roll  = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
            pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
            yaw   = 0.0
        return float(roll), float(pitch), float(yaw)

    def _save_visualization(
        self,
        rgb_msg: Image,
        rgb: np.ndarray,
        mask: np.ndarray,
        rot: np.ndarray,
        trans: np.ndarray,
        conf: float,
        other_instances: Optional[list] = None,
    ):
        if (not self.save_vis) or self.save_vis_every_n <= 0:
            return
        if self.frame_counter % self.save_vis_every_n != 0:
            self.frame_counter += 1
            return
        self.frame_counter += 1

        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 绘制目标分割掩码（蓝色半透明）
        mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
        overlay = vis.copy()
        overlay[mask_2d > 0] = (255, 80, 0)  # BGR 蓝色
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0.0)

        # 绘制位姿坐标轴（X=红, Y=绿, Z=蓝）
        p0 = self._project(trans)
        px = self._project(trans + rot[:, 0] * 0.05)
        py = self._project(trans + rot[:, 1] * 0.05)
        pz = self._project(trans + rot[:, 2] * 0.05)
        if p0 is not None:
            cv2.circle(vis, p0, 4, (0, 255, 255), -1)
            if px is not None:
                cv2.line(vis, p0, px, (0, 0, 255), 2)
            if py is not None:
                cv2.line(vis, p0, py, (0, 255, 0), 2)
            if pz is not None:
                cv2.line(vis, p0, pz, (255, 0, 0), 2)

        # ── HUD：标题 / 偏移向量 / 旋转欧拉角 ───────────────────────────────
        roll, pitch, yaw = self._rot_to_euler_deg(rot)
        hud_lines = [
            f"{self.target_label}(obj={self.obj_id}) conf={conf:.3f}",
            f"T  x={trans[0]:+.3f}  y={trans[1]:+.3f}  z={trans[2]:+.3f} m",
            f"R  roll={roll:+.1f}deg  pitch={pitch:+.1f}deg  yaw={yaw:+.1f}deg",
        ]
        for i, line in enumerate(hud_lines):
            y_pos = 22 + i * 20
            # 黑色描边增强可读性
            cv2.putText(vis, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)
            cv2.putText(vis, line, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        # 绘制其他被检测到的非目标类别（红色边框 + 标签）
        for inst in (other_instances or []):
            inst_mask = inst.get("mask")
            if inst_mask is None:
                continue
            inst_mask_2d = inst_mask[:, :, 0] if inst_mask.ndim == 3 else inst_mask
            coords = cv2.findNonZero((inst_mask_2d > 0).astype(np.uint8))
            if coords is None:
                continue
            bx, by, bw, bh = cv2.boundingRect(coords)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            label_text = f"{inst['label']} {inst['score']:.2f}"
            text_y = max(by - 6, 14)
            cv2.putText(vis, label_text, (bx, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        fname = f"{rgb_msg.header.stamp.sec}_{rgb_msg.header.stamp.nanosec}.png"
        if self.save_video:
            if self.video_writer is None:
                h, w = vis.shape[:2]
                fps = self.video_fps if self.video_fps > 0 else 15.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, fps, (w, h))
                if not self.video_writer.isOpened():
                    self.get_logger().error(f"Failed to open video writer: {self.video_path}")
                    self.video_writer = None
                else:
                    self.get_logger().info(f"Saving visualization video to: {self.video_path}")
            if self.video_writer is not None:
                self.video_writer.write(vis)
        else:
            cv2.imwrite(os.path.join(self.vis_output_dir, fname), vis)

    def _publish_pose(self, rgb_msg: Image, quat: np.ndarray, trans: np.ndarray, rot: np.ndarray):
        pose = PoseStamped()
        pose.header = rgb_msg.header
        pose.pose.position.x = float(trans[0])
        pose.pose.position.y = float(trans[1])
        pose.pose.position.z = float(trans[2])
        pose.pose.orientation.w = float(quat[0])
        pose.pose.orientation.x = float(quat[1])
        pose.pose.orientation.y = float(quat[2])
        pose.pose.orientation.z = float(quat[3])
        self.pose_pub.publish(pose)
        offset = Vector3Stamped()
        offset.header = rgb_msg.header
        offset.vector.x = float(trans[0])
        offset.vector.y = float(trans[1])
        offset.vector.z = float(trans[2])
        self.offset_pub.publish(offset)
        rot_msg = Float64MultiArray()
        rot_msg.data = rot.reshape(-1).astype(float).tolist()
        self.rot_mat_pub.publish(rot_msg)

        pose_result = PoseEstimationResult()
        pose_result.header = rgb_msg.header
        pose_result.frame_id = int(self.pose_result_frame_id)
        pose_result.rotation_matrix = rot.reshape(-1).astype(float).tolist()
        pose_result.translation_vector = trans.reshape(-1).astype(float).tolist()
        self.pose_result_pub.publish(pose_result)
        self.pose_result_frame_id += 1

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.get_logger().info(f"Video saved: {self.video_path}")
        return super().destroy_node()


def main():
    rclpy.init()
    node = DenseFusionRosNodeYcb()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
