source /home/ubuntu/miniconda3/bin/activate onepose
source /opt/ros/jazzy/setup.bash
source /home/ubuntu/tengf/vision-grab/DenseFusion-1/densefusion_ros/install/setup.bash

# ros2 launch orbbec_camera gemini_330_series.launch.py \
#   depth_registration:=true \
#   align_mode:=SW \
#   align_target_stream:=COLOR \
#   color_width:=640 \
#   color_height:=480 \
#   color_fps:=30 \
#   depth_width:=640 \
#   depth_height:=480 \
#   depth_fps:=30

# sleep 5

# ros2 launch densefusion_ros camera_inference.launch.py 