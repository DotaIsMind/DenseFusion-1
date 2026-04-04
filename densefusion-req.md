REVIEW README.md和整个code work space，确保你对DENSEFUSION的代码有清晰的了解。
完成以下任务：
1. 根据eval_linemod.sh的内容，把推理DenseFusion的过程整理到一个python 脚本中，保存为inference_pipeline.py
    i. pipeline使用本地读取图片输入和视频输入两种形式
    ii. 加上Log打印没一张图片前后处理，推理耗时
2. Export pretrained model to onnx model: 我需要使用onnxruntime进行推理，给出DenseFusion模型转换为onnx模型的脚本eport_onnx.py
3. 我最终使用CPU进行推理，所以对torch有依赖的代码，你需要转换为CPU处理的形式。
4. 完成inference_pipeline.py和onnx模型导出后，新增一个benchmark.py文件，对比pth模型和onnx模型的推理性能

Linmod pytorch models path:
/home/data/qrb_ros_simulation_ws/DenseFusion-1/trained_checkpoints/linemod/trained_checkpoints/linemod

你可以使用
source ~/miniconda/bin/activate && conda activate onepose激活虚拟环境
如果你在onepose虚拟环境中遇到问题，记录问题输入为issue-record.md并给出解决方案，完成上述4个目标

