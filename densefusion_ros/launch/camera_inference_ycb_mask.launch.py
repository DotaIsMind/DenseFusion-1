from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pose_onnx_path",
                default_value="/home/data/qrb_ros_simulation_ws/DenseFusion-1/ycb-data-onnx-model/densefusion_ycb_posenet.onnx",
            ),
            DeclareLaunchArgument(
                "refine_onnx_path",
                default_value="/home/data/qrb_ros_simulation_ws/DenseFusion-1/ycb-data-onnx-model/densefusion_ycb_refiner.onnx",
            ),
            DeclareLaunchArgument("rgb_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            DeclareLaunchArgument("mask_topic", default_value="/camera/instance_mask"),
            Node(
                package="densefusion_ros",
                executable="densefusion_ros_node_ycb",
                name="densefusion_ros_node_ycb",
                output="screen",
                parameters=[
                    {
                        "pose_onnx_path": LaunchConfiguration("pose_onnx_path"),
                        "refine_onnx_path": LaunchConfiguration("refine_onnx_path"),
                        # "target_label": "banana",
                        # "obj_id": 11,
                        "target_label": "bottle",
                        "obj_id": 5,
                        "rgb_topic": LaunchConfiguration("rgb_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        # Fixed mask-only mode: non-empty mask_topic disables YOLO branch.
                        "mask_topic": LaunchConfiguration("mask_topic"),
                        "num_points": 1000,
                        "iteration": 2,
                        "cam_fx": 1066.778,
                        "cam_fy": 1067.487,
                        "cam_cx": 312.9869,
                        "cam_cy": 241.3109,
                        # 相机深度图为 16UC1，单位已是毫米，depth_scale=1.0 无需转换
                        "depth_scale": 1.0,
                    }
                ],
            ),
        ]
    )
