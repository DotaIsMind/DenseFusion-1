# Issue Record
# DenseFusion 环境问题记录

```bash
ros2 launch orbbec_camera gemini_330_series.launch.py   depth_registration:=true   align_mode:=SW   align_target_stream:=COLOR   color_width:=640 color_height:=480 color_fps:=30   depth_width:=640 depth_height:=480 depth_fps:=30
```

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


日期: 2026-04-09

## 1) ROS2 启动时报 `ModuleNotFoundError: No module named 'densefusion_ros'`

- 现象:
  - `ros2 launch densefusion_ros local_file_test.launch.py` 后，`file_input_publisher` 和 `densefusion_ros_node` 均因找不到 `densefusion_ros` 模块退出。
- 根因:
  - Python 包目录缺少 `__init__.py`，`find_packages()` 未正确打包模块。
- 处理:
  - 新增:
    - `DenseFusion-1/densefusion_ros/densefusion_ros/__init__.py`
    - `DenseFusion-1/densefusion_ros/densefusion_ros/densefusion_core/__init__.py`
  - 在 `densefusion_core/__init__.py` 中显式导出 `CameraIntrinsics/OBJLIST/OnnxDenseFusion/Yolo11SegOnnx/preprocess_rgbd/quaternion_matrix`。
- 结果:
  - 模块导入问题消失，节点可启动。

## 2) launch 默认参数错误导致文件/模型加载失败

- 现象:
  - `local_file_test.launch.py` 默认路径存在问题，启动时报文件不存在或读取失败。
- 根因:
  - `pose_onnx_path/refine_onnx_path/yolo_seg_onnx_path` 默认值错误（含缺失 `/`、模型指向错误）。
  - `rgb_path/depth_path` 传的是目录而非具体图片文件。
- 处理:
  - 修正 `local_file_test.launch.py` 默认参数:
    - ONNX 路径改为有效绝对路径。
    - YOLO 分割模型改为 `yolo11-seg-model/yolo26n-seg.onnx`。
    - `rgb_path/depth_path` 默认改为 `.../000000.png`。
- 结果:
  - `local_file_test.launch.py` 可正常加载测试输入并运行。

## 3) `target_label` 与 `obj_id` 不联动，导致 YOLO 标签不匹配

- 现象:
  - 默认标签与对象类别可能不一致，YOLO mask 经常不可用并回退到 `depth>0`。
- 处理:
  - 在 `densefusion_ros_node.py` 新增 `OBJ_ID_TO_YOLO_LABEL` 映射。
  - 支持 `target_label=auto`（或空字符串）时按 `obj_id` 自动选标签。
  - `local_file_test.launch.py` 默认 `target_label` 改为 `auto`。
- 结果:
  - 日志可见自动映射生效（如 `obj_id=1 -> bottle`）。

## 4) 可视化增强需求实现

- 需求:
  - 日志输出增加旋转矩阵信息；
  - 增加可视化保存选项，保存 YOLO 分割图和姿态轴图。
- 处理:
  - `densefusion_ros_node.py` 增加:
    - 参数: `save_vis`、`vis_dir`
    - 可视化保存目录: `yolo_seg/`、`axis/`
    - 姿态轴绘制与图像保存逻辑。
  - 日志新增:
    - 旋转矩阵 `R`
    - 欧拉角 `euler_rpy`（roll/pitch/yaw，deg）
  - 默认可视化路径从 `/tmp` 改为项目目录 `./densefusion_vis`。
- 结果:
  - 可视化文件成功落盘，日志包含位姿矩阵与欧拉角。

## 5) 可视化时出现 `IndexError`（mask 与 RGB 尺寸不一致）

- 现象:
  - `overlay[mask > 0]` 处报错，`mask` 尺寸 `480x848`，RGB 为 `720x1280`。
- 根因:
  - YOLO mask 输出分辨率与输入图像分辨率不一致。
- 处理:
  - 在 `_save_visualizations()` 中增加尺寸检查；
  - 尺寸不一致时仅对可视化分支执行最近邻 resize，并打印 warning。
- 结果:
  - 不再因可视化索引崩溃。

## 6) `file_input_publisher` 对 RGB/Depth 分辨率处理策略变更

- 阶段一:
  - 曾加入“发布前将 depth resize 到 RGB 尺寸”逻辑以快速跑通。
- 阶段二（最终）:
  - 按需求取消 resize；
  - 改为严格校验：RGB/Depth 尺寸不一致时直接抛异常并停止。
- 当前行为:
  - 若输入为 `rgb=(720,1280), depth=(480,848)`，会报:
    - `RuntimeError: RGB/Depth resolution mismatch ... Please use aligned streams or matching camera profiles.`

## 7) Orbbec 相机“对齐 + 同分辨率”参数确认

- 在 `OrbbecSDK_ROS2` 中确认了关键参数:
  - `depth_registration`
  - `align_mode`
  - `align_target_stream`
  - `color_width/color_height/color_fps`
  - `depth_width/depth_height/depth_fps`
- 已给出可直接运行命令（示例为 Gemini 330 系列）:
  - 开启 D2C 对齐并设置同分辨率 profile（如 640x480@30）。

