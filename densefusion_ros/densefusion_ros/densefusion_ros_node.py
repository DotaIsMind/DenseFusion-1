import numpy as np
import os
import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from .densefusion_core import CameraIntrinsics, OBJLIST, OnnxDenseFusion, Yolo11SegOnnx, preprocess_rgbd, quaternion_matrix
from .densefusion_core.preprocess import infer_pose_onnx

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


class DenseFusionRosNode(Node):
    def __init__(self):
        super().__init__("densefusion_ros_node")
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("obj_id", 1)
        self.declare_parameter("num_points", 500)
        self.declare_parameter("iteration", 4)
        self.declare_parameter("pose_onnx_path", "")
        self.declare_parameter("refine_onnx_path", "")
        self.declare_parameter(
            "yolo_seg_onnx_path",
            "/home/data/qrb_ros_simulation_ws/DenseFusion-1/yolo11-seg-model/yolo26n-seg.onnx",
        )
        # "auto": choose a recommended YOLO label from obj_id.
        self.declare_parameter("target_label", "auto")
        self.declare_parameter("yolo_score_th", 0.25)
        self.declare_parameter("yolo_mask_th", 0.5)
        self.declare_parameter("cam_fx", 572.41140)
        self.declare_parameter("cam_fy", 573.57043)
        self.declare_parameter("cam_cx", 325.26110)
        self.declare_parameter("cam_cy", 242.04899)
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

        if self.target_label.strip().lower() in ("", "auto"):
            auto_label = OBJ_ID_TO_YOLO_LABEL.get(self.obj_id, "bottle")
            self.target_label = auto_label
            self.get_logger().info(
                f"Auto target_label enabled: obj_id={self.obj_id} -> target_label='{self.target_label}'"
            )
        else:
            self.get_logger().info(
                f"Using manual target_label='{self.target_label}' for obj_id={self.obj_id}"
            )

        self.obj_index = OBJLIST.index(self.obj_id)
        self.runner = OnnxDenseFusion(self.pose_onnx_path, self.refine_onnx_path)
        self.yolo_seg = Yolo11SegOnnx(
            self.yolo_seg_onnx_path,
            score_th=self.yolo_score_th,
            mask_th=self.yolo_mask_th,
        )
        self.bridge = CvBridge()
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

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self._on_rgbd)
        self.get_logger().info(f"Subscribed RGB={self.rgb_topic}, Depth={self.depth_topic}")
        self.get_logger().info(f"Using YOLO ONNX segmentation model: {self.yolo_seg_onnx_path}")

    def _to_depth_mm(self, depth_msg: Image) -> np.ndarray:
        if depth_msg.encoding == "16UC1":
            depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            return np.array(depth_mm, dtype=np.float32)
        depth_m = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_m = np.array(depth_m, dtype=np.float32)
        return depth_m * self.depth_scale

    def _on_rgbd(self, rgb_msg: Image, depth_msg: Image):
        rgb_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        rgb = rgb_bgr[:, :, ::-1]
        depth = self._to_depth_mm(depth_msg)
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
