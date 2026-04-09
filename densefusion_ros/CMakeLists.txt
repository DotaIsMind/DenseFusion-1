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

# Install Python source files as symlinks so code changes take effect immediately
# without a full rebuild. Using install(CODE) + cmake -E create_symlink because
# install(DIRECTORY) always copies (symlink-install has no effect on it), and
# ament_python_install_package() conflicts with rosidl-generated "densefusion_ros"
# Python modules that share the same package name.
install(CODE "
  set(_src_dir \"${CMAKE_CURRENT_SOURCE_DIR}/densefusion_ros\")
  set(_dst_dir \"\${CMAKE_INSTALL_PREFIX}/${PYTHON_INSTALL_DIR}/densefusion_app\")
  file(GLOB_RECURSE _py_files
    RELATIVE \"${CMAKE_CURRENT_SOURCE_DIR}/densefusion_ros\"
    \"${CMAKE_CURRENT_SOURCE_DIR}/densefusion_ros/*.py\")
  foreach(_rel_path \${_py_files})
    set(_src \"\${_src_dir}/\${_rel_path}\")
    set(_dst \"\${_dst_dir}/\${_rel_path}\")
    get_filename_component(_dst_parent \"\${_dst}\" DIRECTORY)
    execute_process(COMMAND \${CMAKE_COMMAND} -E make_directory \"\${_dst_parent}\")
    execute_process(COMMAND \${CMAKE_COMMAND} -E remove -f \"\${_dst}\")
    execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink \"\${_src}\" \"\${_dst}\")
  endforeach()
")

install(
  PROGRAMS
    densefusion_ros_node
    densefusion_ros_node_ycb
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
