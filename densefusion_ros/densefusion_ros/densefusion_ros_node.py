import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from .densefusion_core import CameraIntrinsics, OBJLIST, OnnxDenseFusion, preprocess_rgbd, quaternion_matrix
from .densefusion_core.preprocess import infer_pose_onnx


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
        self.declare_parameter("cam_fx", 572.41140)
        self.declare_parameter("cam_fy", 573.57043)
        self.declare_parameter("cam_cx", 325.26110)
        self.declare_parameter("cam_cy", 242.04899)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("input_h", 80)
        self.declare_parameter("input_w", 80)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.obj_id = int(self.get_parameter("obj_id").value)
        self.num_points = int(self.get_parameter("num_points").value)
        self.iteration = int(self.get_parameter("iteration").value)
        self.pose_onnx_path = self.get_parameter("pose_onnx_path").value
        self.refine_onnx_path = self.get_parameter("refine_onnx_path").value
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.input_h = int(self.get_parameter("input_h").value)
        self.input_w = int(self.get_parameter("input_w").value)
        self.intr = CameraIntrinsics(
            cam_fx=float(self.get_parameter("cam_fx").value),
            cam_fy=float(self.get_parameter("cam_fy").value),
            cam_cx=float(self.get_parameter("cam_cx").value),
            cam_cy=float(self.get_parameter("cam_cy").value),
        )

        if self.obj_id not in OBJLIST:
            raise ValueError(f"obj_id must be one of {OBJLIST}, got {self.obj_id}")
        if not self.pose_onnx_path or not self.refine_onnx_path:
            raise ValueError("pose_onnx_path and refine_onnx_path are required parameters")

        self.obj_index = OBJLIST.index(self.obj_id)
        self.runner = OnnxDenseFusion(self.pose_onnx_path, self.refine_onnx_path)
        self.bridge = CvBridge()

        self.pose_pub = self.create_publisher(PoseStamped, "/pose_stamp", 10)
        self.offset_pub = self.create_publisher(Vector3Stamped, "/pose_stamp_offset", 10)
        self.rot_mat_pub = self.create_publisher(Float64MultiArray, "/pose_stamp_rotation_matrix", 10)

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic)
        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self._on_rgbd)
        self.get_logger().info(f"Subscribed RGB={self.rgb_topic}, Depth={self.depth_topic}")

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
        mask = (depth > 0).astype(np.uint8) * 255

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
        self.get_logger().info(
            f"Published /pose_stamp | conf={conf:.4f} t=({trans[0]:.4f},{trans[1]:.4f},{trans[2]:.4f})"
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
