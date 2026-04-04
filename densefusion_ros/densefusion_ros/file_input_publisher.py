import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class FileInputPublisher(Node):
    def __init__(self):
        super().__init__("file_input_publisher")
        self.declare_parameter("rgb_path", "")
        self.declare_parameter("depth_path", "")
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("publish_hz", 5.0)

        self.rgb_path = self.get_parameter("rgb_path").value
        self.depth_path = self.get_parameter("depth_path").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        if not self.rgb_path or not self.depth_path:
            raise ValueError("rgb_path and depth_path are required for file input publisher")

        self.bridge = CvBridge()
        self.rgb_pub = self.create_publisher(Image, self.rgb_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, 10)
        self.timer = self.create_timer(max(0.01, 1.0 / self.publish_hz), self._tick)

        self.rgb = cv2.imread(self.rgb_path, cv2.IMREAD_COLOR)
        self.depth = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED)
        if self.rgb is None or self.depth is None:
            raise RuntimeError(f"Failed to load rgb/depth from {self.rgb_path} / {self.depth_path}")
        self.get_logger().info(f"Publishing test files to {self.rgb_topic} and {self.depth_topic}")

    def _tick(self):
        stamp = self.get_clock().now().to_msg()
        rgb_msg = self.bridge.cv2_to_imgmsg(self.rgb, encoding="bgr8")
        depth_msg = self.bridge.cv2_to_imgmsg(self.depth, encoding="16UC1")
        rgb_msg.header.stamp = stamp
        depth_msg.header.stamp = stamp
        rgb_msg.header.frame_id = "camera_color_frame"
        depth_msg.header.frame_id = "camera_depth_frame"
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)


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
