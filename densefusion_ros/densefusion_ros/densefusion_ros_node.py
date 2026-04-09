import numpy as np
import os
import sys
import cv2
import rclpy
from pathlib import Path
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
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

    # Also reuse existing sys.path entries that already contain generated messages.
    for path in list(sys.path):
        p = Path(path)
        if (p / "densefusion_ros" / "msg").is_dir():
            generated_sites.append(str(p))

    # Move generated message site-packages ahead of workspace source dirs to avoid
    # importing the source Python package "densefusion_ros" (which has no .msg).
    for site in reversed(list(dict.fromkeys(generated_sites))):
        if site in sys.path:
            sys.path.remove(site)
        sys.path.insert(0, site)

    # If a shadowing source package has already been imported, discard it.
    loaded_pkg = sys.modules.get("densefusion_ros")
    loaded_file = str(getattr(loaded_pkg, "__file__", ""))
    if loaded_pkg is not None and "/site-packages/" not in loaded_file:
        sys.modules.pop("densefusion_ros", None)
        sys.modules.pop("densefusion_ros.msg", None)

    from densefusion_ros.msg import PoseEstimationResult

    # Ensure generated rosidl type support is loaded before create_publisher.
    PoseEstimationResult.__class__.__import_type_support__()
    if PoseEstimationResult.__class__._TYPE_SUPPORT is None:
        raise RuntimeError(
            "PoseEstimationResult type support is unavailable. "
            "Please rebuild and source this workspace: "
            "`colcon build --symlink-install --packages-select densefusion_ros --base-paths densefusion_ros` "
            "then `source install/setup.bash`."
        )
    return PoseEstimationResult


PoseEstimationResult = _load_pose_result_msg_type()

from .densefusion_core import CameraIntrinsics, OBJLIST, OnnxDenseFusion, Yolo11SegOnnx, preprocess_rgbd, quaternion_matrix
from .densefusion_core.preprocess import infer_pose_onnx

"""
OBJ_ID_TO_YOLO_LABEL = {
    1: "bottle",
    2: "cup",
    4: "camera",
    5: "bottle",
    6: "cat",
    8: "drill",
    9: "bird",
    10: "box",
    11: "bottle",
    12: "cup",
    13: "iron",
    14: "lamp",
    15: "cell phone",
}
"""
CUP_OBJ_ID = 2
CUP_LABEL = "cup"


class DenseFusionRosNode(Node):
    def __init__(self):
        super().__init__("densefusion_ros_node")
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("obj_id", CUP_OBJ_ID)
        self.declare_parameter("num_points", 500)
        self.declare_parameter("iteration", 4)
        self.declare_parameter("pose_onnx_path", "")
        self.declare_parameter("refine_onnx_path", "")
        self.declare_parameter(
            "yolo_seg_onnx_path",
            "/home/data/qrb_ros_simulation_ws/DenseFusion-1/yolo11-seg-model/yolo26n-seg.onnx",
        )
        self.declare_parameter("target_label", CUP_LABEL)
        self.declare_parameter("yolo_score_th", 0.25)
        self.declare_parameter("yolo_mask_th", 0.5)
        self.declare_parameter("cam_fx", 461.07720947265625)
        self.declare_parameter("cam_fy", 461.29638671875)
        self.declare_parameter("cam_cx", 318.0372009277344)
        self.declare_parameter("cam_cy", 236.3270721435547)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("input_h", 80)
        self.declare_parameter("input_w", 80)
        self.declare_parameter("save_vis", False)
        self.declare_parameter("vis_dir", "./densefusion_vis")

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.obj_id = int(self.get_parameter("obj_id").value)
        self.num_points = int(self.get_parameter("num_points").value)
        self.iteration = int(self.get_parameter("iteration").value)
        self.pose_onnx_path = self.get_parameter("pose_onnx_path").value
        self.refine_onnx_path = self.get_parameter("refine_onnx_path").value
        self.yolo_seg_onnx_path = self.get_parameter("yolo_seg_onnx_path").value
        self.target_label = str(self.get_parameter("target_label").value)
        self.yolo_score_th = float(self.get_parameter("yolo_score_th").value)
        self.yolo_mask_th = float(self.get_parameter("yolo_mask_th").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.input_h = int(self.get_parameter("input_h").value)
        self.input_w = int(self.get_parameter("input_w").value)
        self.save_vis = bool(self.get_parameter("save_vis").value)
        self.vis_dir = os.path.abspath(str(self.get_parameter("vis_dir").value))
        self.intr = CameraIntrinsics(
            cam_fx=float(self.get_parameter("cam_fx").value),
            cam_fy=float(self.get_parameter("cam_fy").value),
            cam_cx=float(self.get_parameter("cam_cx").value),
            cam_cy=float(self.get_parameter("cam_cy").value),
        )

        if self.obj_id not in OBJLIST:
            raise ValueError(f"obj_id must be one of {OBJLIST}, got {self.obj_id}")
        if not self.pose_onnx_path or not self.refine_onnx_path or not self.yolo_seg_onnx_path:
            raise ValueError("pose_onnx_path, refine_onnx_path and yolo_seg_onnx_path are required parameters")
        if not self.yolo_seg_onnx_path.endswith(".onnx"):
            raise ValueError(f"yolo_seg_onnx_path must be an ONNX file, got: {self.yolo_seg_onnx_path}")
        if not os.path.exists(self.yolo_seg_onnx_path):
            raise FileNotFoundError(f"YOLO ONNX model not found: {self.yolo_seg_onnx_path}")

        if self.obj_id != CUP_OBJ_ID:
            self.get_logger().warn(
                f"Only CUP is supported in this node. Override obj_id {self.obj_id} -> {CUP_OBJ_ID}."
            )
            self.obj_id = CUP_OBJ_ID
        if self.target_label.strip().lower() != CUP_LABEL:
            self.get_logger().warn(
                f"Only CUP is supported in this node. Override target_label '{self.target_label}' -> '{CUP_LABEL}'."
            )
            self.target_label = CUP_LABEL
        self.get_logger().info(f"CUP-only mode enabled: obj_id={self.obj_id}, target_label='{self.target_label}'")

        self.obj_index = OBJLIST.index(self.obj_id)
        self.runner = OnnxDenseFusion(self.pose_onnx_path, self.refine_onnx_path)
        self.yolo_seg = Yolo11SegOnnx(
            self.yolo_seg_onnx_path,
            score_th=self.yolo_score_th,
            mask_th=self.yolo_mask_th,
        )
        self.vis_index = 0
        if self.save_vis:
            self.seg_vis_dir = os.path.join(self.vis_dir, "yolo_seg")
            self.axis_vis_dir = os.path.join(self.vis_dir, "axis")
            os.makedirs(self.seg_vis_dir, exist_ok=True)
            os.makedirs(self.axis_vis_dir, exist_ok=True)
            self.get_logger().info(
                f"Visualization saving enabled: seg={self.seg_vis_dir}, axis={self.axis_vis_dir}"
            )

        self.pose_pub = self.create_publisher(PoseStamped, "/pose_stamp", 10)
        self.offset_pub = self.create_publisher(Vector3Stamped, "/pose_stamp_offset", 10)
        self.rot_mat_pub = self.create_publisher(Float64MultiArray, "/pose_stamp_rotation_matrix", 10)
        self.pose_result_pub = self.create_publisher(PoseEstimationResult, "/pose_estimation_result", 10)
        self.pose_result_frame_id = 0

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self._on_rgbd)
        self.get_logger().info(f"Subscribed RGB={self.rgb_topic}, Depth={self.depth_topic}")
        self.get_logger().info(f"Using YOLO ONNX segmentation model: {self.yolo_seg_onnx_path}")

    @staticmethod
    def _apply_endianness(arr: np.ndarray, msg: Image) -> np.ndarray:
        msg_is_big = bool(msg.is_bigendian)
        host_is_big = sys.byteorder == "big"
        if msg_is_big != host_is_big:
            return arr.byteswap().view(arr.dtype.newbyteorder("="))
        return arr

    def _to_bgr(self, rgb_msg: Image) -> np.ndarray:
        encoding = str(rgb_msg.encoding).lower()
        raw = np.frombuffer(rgb_msg.data, dtype=np.uint8)
        row_stride = int(rgb_msg.step)
        if row_stride <= 0:
            raise ValueError(f"Invalid rgb step={row_stride}")
        if encoding in ("bgr8", "rgb8"):
            needed = int(rgb_msg.width) * 3
            img = raw.reshape(int(rgb_msg.height), row_stride)[:, :needed].reshape(int(rgb_msg.height), int(rgb_msg.width), 3)
            if encoding == "rgb8":
                img = img[:, :, ::-1]
            return np.ascontiguousarray(img)
        if encoding in ("bgra8", "rgba8"):
            needed = int(rgb_msg.width) * 4
            img = raw.reshape(int(rgb_msg.height), row_stride)[:, :needed].reshape(int(rgb_msg.height), int(rgb_msg.width), 4)
            if encoding == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return np.ascontiguousarray(img)
        raise ValueError(f"Unsupported rgb encoding: {rgb_msg.encoding}")

    def _to_depth_mm(self, depth_msg: Image) -> np.ndarray:
        encoding = str(depth_msg.encoding).lower()
        h = int(depth_msg.height)
        w = int(depth_msg.width)
        if encoding in ("16uc1", "mono16"):
            raw = np.frombuffer(depth_msg.data, dtype=np.uint16)
            depth = raw.reshape(h, int(depth_msg.step) // 2)[:, :w]
            depth = self._apply_endianness(depth, depth_msg)
            return depth.astype(np.float32)
        if encoding in ("32fc1", "32fc"):
            raw = np.frombuffer(depth_msg.data, dtype=np.float32)
            depth_m = raw.reshape(h, int(depth_msg.step) // 4)[:, :w]
            depth_m = self._apply_endianness(depth_m, depth_msg)
            return depth_m.astype(np.float32) * self.depth_scale
        raise ValueError(f"Unsupported depth encoding: {depth_msg.encoding}")

    def _on_rgbd(self, rgb_msg: Image, depth_msg: Image):
        try:
            rgb_bgr = self._to_bgr(rgb_msg)
            depth = self._to_depth_mm(depth_msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode image messages: {exc}")
            return
        rgb = rgb_bgr[:, :, ::-1]
        if rgb.shape[:2] != depth.shape[:2]:
            self.get_logger().warn(
                f"RGB/Depth resolution mismatch: rgb={rgb.shape[:2]} depth={depth.shape[:2]}; frame skipped."
            )
            return
        mask = self.yolo_seg.infer_mask(rgb, self.target_label)
        mask_source = "yolo"
        if mask is None:
            self.get_logger().warn(f"YOLO-Seg mask unavailable for label={self.target_label}, fallback to depth>0 mask.")
            mask = (depth > 0).astype(np.uint8) * 255
            mask_source = "depth_fallback"

        data = preprocess_rgbd(
            rgb,
            depth,
            mask,
            self.obj_index,
            self.num_points,
            self.intr,
            input_h=self.input_h,
            input_w=self.input_w,
        )
        if data is None:
            self.get_logger().warn("No valid points from depth mask; frame skipped.")
            return

        quat, trans, conf = infer_pose_onnx(self.runner, data, self.iteration, self.num_points)
        rot = quaternion_matrix(quat)[:3, :3]
        self._publish_pose(rgb_msg, quat, trans, rot)
        if self.save_vis:
            self._save_visualizations(rgb_bgr, mask, rot, trans, conf, mask_source)
        euler_deg = self._rotation_matrix_to_euler_zyx_deg(rot)
        rot_str = np.array2string(rot, precision=4, suppress_small=True, separator=", ").replace("\n", "")
        euler_str = f"(roll={euler_deg[0]:.2f}, pitch={euler_deg[1]:.2f}, yaw={euler_deg[2]:.2f})deg"
        self.get_logger().info(
            f"Published /pose_stamp | conf={conf:.4f} t=({trans[0]:.4f},{trans[1]:.4f},{trans[2]:.4f}) R={rot_str} euler_rpy={euler_str}"
        )

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

    def _project_point(self, pt3: np.ndarray):
        x, y, z = float(pt3[0]), float(pt3[1]), float(pt3[2])
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(z) or z <= 1e-6:
            return None
        u = int(round((x * self.intr.cam_fx) / z + self.intr.cam_cx))
        v = int(round((y * self.intr.cam_fy) / z + self.intr.cam_cy))
        return (u, v)

    @staticmethod
    def _rotation_matrix_to_euler_zyx_deg(rot: np.ndarray) -> np.ndarray:
        # Returns roll(x), pitch(y), yaw(z) in degrees for ZYX convention.
        sy = float(np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
        singular = sy < 1e-6
        if not singular:
            roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
            pitch = float(np.arctan2(-rot[2, 0], sy))
            yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        else:
            # Gimbal-lock fallback.
            roll = float(np.arctan2(-rot[1, 2], rot[1, 1]))
            pitch = float(np.arctan2(-rot[2, 0], sy))
            yaw = 0.0
        return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))

    def _draw_axis_vis(self, rgb_bgr: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
        vis = rgb_bgr.copy()
        axis_len = 0.05
        origin = trans
        x_end = origin + rot[:, 0] * axis_len
        y_end = origin + rot[:, 1] * axis_len
        z_end = origin + rot[:, 2] * axis_len
        p0 = self._project_point(origin)
        px = self._project_point(x_end)
        py = self._project_point(y_end)
        pz = self._project_point(z_end)
        if p0 is None:
            return vis
        cv2.circle(vis, p0, 4, (0, 255, 255), -1)
        if px is not None:
            cv2.line(vis, p0, px, (0, 0, 255), 2)
        if py is not None:
            cv2.line(vis, p0, py, (0, 255, 0), 2)
        if pz is not None:
            cv2.line(vis, p0, pz, (255, 0, 0), 2)
        return vis

    def _save_visualizations(
        self,
        rgb_bgr: np.ndarray,
        mask: np.ndarray,
        rot: np.ndarray,
        trans: np.ndarray,
        conf: float,
        mask_source: str,
    ) -> None:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        h, w = rgb_bgr.shape[:2]
        if mask.shape[:2] != (h, w):
            self.get_logger().warn(
                f"Vis mask size {mask.shape[:2]} != rgb size {(h, w)}; resizing mask for visualization only."
            )
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        seg_vis = rgb_bgr.copy()
        overlay = seg_vis.copy()
        overlay[mask > 0] = (0, 165, 255)
        seg_vis = cv2.addWeighted(overlay, 0.45, seg_vis, 0.55, 0.0)
        cv2.putText(
            seg_vis,
            f"label={self.target_label} source={mask_source}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        axis_vis = self._draw_axis_vis(rgb_bgr, rot, trans)
        cv2.putText(
            axis_vis,
            f"conf={conf:.4f}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        sid = f"{self.vis_index:06d}"
        seg_path = os.path.join(self.seg_vis_dir, f"{sid}.png")
        axis_path = os.path.join(self.axis_vis_dir, f"{sid}.png")
        cv2.imwrite(seg_path, seg_vis)
        cv2.imwrite(axis_path, axis_vis)
        self.vis_index += 1


def main():
    rclpy.init()
    node = DenseFusionRosNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
