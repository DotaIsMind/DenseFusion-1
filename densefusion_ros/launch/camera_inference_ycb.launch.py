from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pose_onnx_path",
                default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/ycb-data-onnx-model/densefusion_ycb_posenet.onnx",
            ),
            DeclareLaunchArgument(
                "refine_onnx_path",
                default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/ycb-data-onnx-model/densefusion_ycb_refiner.onnx",
            ),
            DeclareLaunchArgument(
                "yolo_seg_onnx_path",
                default_value="/home/ubuntu/tengf/vision-grab/DenseFusion-1/yolo11-seg-model/yolo26n-seg.onnx",
            ),
            DeclareLaunchArgument("rgb_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
            DeclareLaunchArgument("mask_topic", default_value=""),
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
                        # "target_label": "banana",
                        # "obj_id": 11,
                        "target_label": "bottle",
                        "obj_id": 5,
                        "rgb_topic": LaunchConfiguration("rgb_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "mask_topic": LaunchConfiguration("mask_topic"),
                        "num_points": 1000,
                        "iteration": 2,
                        "cam_fx": 461.07720947265625,
                        "cam_fy": 461.29638671875,
                        "cam_cx": 318.0372009277344,
                        "cam_cy": 236.3270721435547,
                        # 相机深度图为 16UC1，单位已是毫米，depth_scale=1.0 无需转换
                        "depth_scale": 1.0,
                    }
                ],
            ),
        ]
    )
