cmake_minimum_required(VERSION 3.8)
project(densefusion_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseEstimationResult.msg"
  DEPENDENCIES std_msgs
)

# Install Python node package manually to avoid target conflicts between
# ament_python_install_package() and rosidl-generated "densefusion_ros" Python modules.
install(
  DIRECTORY densefusion_ros/
  DESTINATION ${PYTHON_INSTALL_DIR}/densefusion_app
  FILES_MATCHING PATTERN "*.py"
)

install(
  PROGRAMS
    densefusion_ros_node
    file_input_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
