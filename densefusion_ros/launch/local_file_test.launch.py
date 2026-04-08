from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("pose_onnx_path"),
            DeclareLaunchArgument("refine_onnx_path"),
            DeclareLaunchArgument("yolo_seg_onnx_path"),
            DeclareLaunchArgument("target_label", default_value="cup"),
            DeclareLaunchArgument("rgb_path"),
            DeclareLaunchArgument("depth_path"),
            DeclareLaunchArgument("obj_id", default_value="1"),
            Node(
                package="densefusion_ros",
                executable="file_input_publisher",
                name="file_input_publisher",
                output="screen",
                parameters=[
                    {
                        "rgb_path": LaunchConfiguration("rgb_path"),
                        "depth_path": LaunchConfiguration("depth_path"),
                        "rgb_topic": "/camera/color/image_raw",
                        "depth_topic": "/camera/depth/image_raw",
                        "publish_hz": 5.0,
                    }
                ],
            ),
            Node(
                package="densefusion_ros",
                executable="densefusion_ros_node",
                name="densefusion_ros_node",
                output="screen",
                parameters=[
                    {
                        "pose_onnx_path": LaunchConfiguration("pose_onnx_path"),
                        "refine_onnx_path": LaunchConfiguration("refine_onnx_path"),
                        "yolo_seg_onnx_path": LaunchConfiguration("yolo_seg_onnx_path"),
                        "target_label": LaunchConfiguration("target_label"),
                        "obj_id": LaunchConfiguration("obj_id"),
                        "rgb_topic": "/camera/color/image_raw",
                        "depth_topic": "/camera/depth/image_raw",
                    }
                ],
            ),
        ]
    )
