REVIEW inference_pipeline.py的代码，确保你对DENSEFUSION的代码有清晰的了解。
完成以下任务：
1. 把inference_pipeline.py修改为订阅/camera/color/raw_image和/camera/depth/image作为RGB-D的输入，使用ROS2 Jazzy
2. 完成推理后发布标准的/pose_stamp结果，包含stamp，偏移向量和旋转矩阵
3. 要求ros package能够单独编译和运行，所以inference的所有依赖你都需要打包到ros package中
4. 编写两个launch file，一个是从local file加载图像作为输入测试ros node，另一个是camera launch file，订阅相机/camera/color/raw_image和/camera/depth/image作为输入
