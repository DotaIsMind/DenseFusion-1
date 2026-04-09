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
            DeclareLaunchArgument(
                "yolo_seg_onnx_path",
                default_value="/home/data/qrb_ros_simulation_ws/DenseFusion-1/yolo11-seg-model/yolo26n-seg.onnx",
            ),
            # 目录模式：指向包含 rgb/ 和 depth/ 子目录的 scene 根目录
            DeclareLaunchArgument(
                "scene_dir",
                default_value="/home/data/qrb_ros_simulation_ws/DenseFusion-1/datasets/ycb-test-data/test~left_pbr/000045",
            ),
            DeclareLaunchArgument("mask_topic", default_value=""),
            Node(
                package="densefusion_ros",
                executable="file_input_publisher",
                name="file_input_publisher",
                output="screen",
                parameters=[
                    {
                        "scene_dir": LaunchConfiguration("scene_dir"),
                        "rgb_topic": "/camera/color/image_raw",
                        "depth_topic": "/camera/depth/image_raw",
                        "publish_hz": 5.0,
                        "loop": False,
                    }
                ],
            ),
            Node(
                package="densefusion_ros",
                executable="densefusion_ros_node_ycb",
                name="densefusion_ros_node_ycb",
                output="screen",
                parameters=[
                    {
                        "pose_onnx_path": LaunchConfiguration("pose_onnx_path"),
                        "refine_onnx_path": LaunchConfiguration("refine_onnx_path"),
                        "yolo_seg_onnx_path": LaunchConfiguration("yolo_seg_onnx_path"),
                        "target_label": "bottle",
                        "obj_id": 5,  # 006_mustard_bottle
                        "mask_topic": LaunchConfiguration("mask_topic"),
                        "rgb_topic": "/camera/color/image_raw",
                        "depth_topic": "/camera/depth/image_raw",
                        "num_points": 1000,
                        "iteration": 2,
                        "cam_fx": 1066.778,
                        "cam_fy": 1067.487,
                        "cam_cx": 312.9869,
                        "cam_cy": 241.3109,
                        # YCB 深度图单位为 0.1mm（cam_scale=10000），乘以 0.1 转换为毫米
                        "depth_scale": 0.1,
                    }
                ],
            ),
        ]
    )
