# DenseFusion 环境问题记录

## Issue 1: 提供的 onepose 激活命令路径不存在

- 问题输入:
  - `source ~/miniconda/bin/activate && conda activate onepose`
- 报错:
  - `/home/data/miniconda/bin/activate: No such file or directory`
- 原因:
  - 当前机器安装路径是 `~/miniconda3`，不是 `~/miniconda`。
- 解决方案:
  - 使用 conda 官方推荐方式初始化 shell 后再激活环境:
  - `source ~/miniconda3/etc/profile.d/conda.sh && conda activate onepose`

## Issue 2: 未激活 onepose 时缺少 torch

- 问题输入:
  - `python3 inference_pipeline.py --help`
- 报错:
  - `ModuleNotFoundError: No module named 'torch'`
- 原因:
  - 当前 shell 不在 onepose 环境中，默认 Python 无 PyTorch。
- 解决方案:
  - 先执行环境激活命令:
  - `source ~/miniconda3/etc/profile.d/conda.sh && conda activate onepose`
  - 再执行推理、导出和 benchmark 脚本。

## Issue 3: ONNX 导出缺少 onnxscript 依赖

- 问题输入:
  - `python3 export_onnx.py ...`
- 报错:
  - `ModuleNotFoundError: No module named 'onnxscript'`
- 原因:
  - 当前 PyTorch ONNX 导出链路依赖 `onnxscript`，环境中未安装。
- 解决方案:
  - 安装依赖后重试导出:
  - `pip install onnxscript`

## Issue 4: DataParallel 导致 CPU 导出/推理冲突

- 问题输入:
  - 导出和 benchmark 过程中调用 `PoseNet` / `PoseRefineNet`
- 报错:
  - `module must have its parameters and buffers on device cuda:0 ... but found ... cpu`
  - 或索引/特征张量设备不一致（cpu vs cuda）
- 原因:
  - `lib/network.py` 中 `ModifiedResnet` 默认包装 `nn.DataParallel`，模型参数初始在 `cuda:0` 语义下运行，不适配 CPU-only 流程。
- 解决方案:
  - 在脚本中显式 `to("cpu")`，并递归移除 `DataParallel` 包装（unwrap），再执行导出/benchmark。
  - 已在 `inference_pipeline.py` 与 `export_onnx.py` 中实现 CPU-safe 处理。

## Issue 5: 动态轴 ONNX 导出失败（num_points 被推断为静态）

- 问题输入:
  - 使用 `dynamo=True` 且开启 `dynamic_axes` 导出 PoseNet
- 报错:
  - `Received user-specified dim hint Dim.DYNAMIC ... inferred a static shape of 500`
- 原因:
  - 模型路径里 `num_points` 在该配置下被图追踪为固定值（500），与动态轴约束冲突。
- 解决方案:
  - 去掉动态轴，改为静态形状导出（固定 `num_points=500`）。

## Issue 6: legacy 导出器在动态空间尺寸上不支持 adaptive_avg_pool2d

- 问题输入:
  - 使用 `dynamo=False`（legacy TorchScript exporter）导出 PoseNet
- 报错:
  - `Unsupported: ONNX export of operator adaptive_avg_pool2d, input size not accessible`
- 原因:
  - PSPNet 模块包含 `adaptive_avg_pool2d`，legacy 导出器对动态空间维处理能力不足。
- 解决方案:
  - 使用 `dynamo=True` 新导出器并固定导出输入尺寸（例如 `input_h=80, input_w=80`）。
  - 当前可用导出命令示例:
  - `python3 export_onnx.py --model ... --refine_model ... --input_h 80 --input_w 80`
