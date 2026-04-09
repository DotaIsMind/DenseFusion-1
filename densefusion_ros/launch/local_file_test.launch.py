from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("pose_onnx_path", default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/densefusion_posenet.onnx"),
            DeclareLaunchArgument("refine_onnx_path", default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/densefusion_refiner.onnx"),
            DeclareLaunchArgument("yolo_seg_onnx_path", default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/yolo11-seg-model/yolo26n-seg.onnx"),
            DeclareLaunchArgument("target_label", default_value="auto"),
            DeclareLaunchArgument("save_vis", default_value="true"),
            DeclareLaunchArgument(
                "vis_dir",
                default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/densefusion_ros/densefusion_vis",
            ),
            DeclareLaunchArgument("rgb_path", default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/densefusion_ros/Linemod_preprocessed/data/01/rgb/000170.png"),
            DeclareLaunchArgument("depth_path", default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/densefusion_ros/Linemod_preprocessed/data/01/depth/000170.png"),
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
                        "save_vis": LaunchConfiguration("save_vis"),
                        "vis_dir": LaunchConfiguration("vis_dir"),
                        "rgb_topic": "/camera/color/image_raw",
                        "depth_topic": "/camera/depth/image_raw",
                    }
                ],
            ),
        ]
    )
