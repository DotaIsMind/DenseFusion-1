# densefusion_ros

ROS2 Jazzy DenseFusion ONNX CPU inference package.

## Build

```bash
cd /home/data/qrb_ros_simulation_ws/DenseFusion-1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate onepose
colcon build --packages-select densefusion_ros --base-paths densefusion_ros
```

## Source Environment

```bash
cd /home/data/qrb_ros_simulation_ws/DenseFusion-1
source install/setup.bash
```

## Launch: Camera Topics Input

订阅：
- `/camera/color/image_raw`
- `/camera/depth/image_raw`

```bash
ros2 launch densefusion_ros camera_inference.launch.py \
  pose_onnx_path:=/home/data/qrb_ros_simulation_ws/DenseFusion-1/densefusion_posenet.onnx \
  refine_onnx_path:=/home/data/qrb_ros_simulation_ws/DenseFusion-1/densefusion_refiner.onnx \
  fastsam_onnx_path:=/home/data/qrb_ros_simulation_ws/DenseFusion-1/fast-sam-model/fast-sam-x.onnx \
  obj_id:=1
```

## Launch: Local File Test Input

```bash
ros2 launch densefusion_ros local_file_test.launch.py \
  pose_onnx_path:=/home/data/qrb_ros_simulation_ws/DenseFusion-1/densefusion_posenet.onnx \
  refine_onnx_path:=/home/data/qrb_ros_simulation_ws/DenseFusion-1/densefusion_refiner.onnx \
  fastsam_onnx_path:=/home/data/qrb_ros_simulation_ws/DenseFusion-1/fast-sam-model/fast-sam-x.onnx \
  rgb_path:=/home/data/qrb_ros_simulation_ws/DenseFusion-1/datasets/Linemod_preprocessed/Linemod_preprocessed/data/01/rgb/0000.png \
  depth_path:=/home/data/qrb_ros_simulation_ws/DenseFusion-1/datasets/Linemod_preprocessed/Linemod_preprocessed/data/01/depth/0000.png \
  obj_id:=1
```

## Published Topics

- `/pose_stamp` (`geometry_msgs/PoseStamped`)
- `/pose_stamp_offset` (`geometry_msgs/Vector3Stamped`)
- `/pose_stamp_rotation_matrix` (`std_msgs/Float64MultiArray`)
