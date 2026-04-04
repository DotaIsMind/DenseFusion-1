from setuptools import find_packages, setup

package_name = "densefusion_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/camera_inference.launch.py", "launch/local_file_test.launch.py"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python",
        "onnxruntime",
    ],
    zip_safe=True,
    maintainer="densefusion_user",
    maintainer_email="user@example.com",
    description="ROS2 Jazzy DenseFusion RGB-D inference package",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "densefusion_ros_node = densefusion_ros.densefusion_ros_node:main",
            "file_input_publisher = densefusion_ros.file_input_publisher:main",
        ],
    },
)
