# DenseFusion ROS2 问题记录

## Issue 1: `package.xml` 出现重复内容块

- 问题现象:
  - 初次写入 ROS 包元数据后，`densefusion_ros/package.xml` 中出现了两份重复的 XML 段。
- 影响:
  - 会导致后续维护困难，且在某些工具链下可能触发解析异常。
- 解决方案:
  - 删除重复段，保留单一 `<package>` 定义，重新执行构建验证。

## Issue 2: DenseFusion 原始流程依赖实例 mask，但相机输入只有 RGB + Depth

- 问题现象:
  - 任务要求订阅 `/camera/color/raw_image` 和 `/camera/depth/image`，未提供单独的语义/实例分割结果。
  - DenseFusion 原推理流程默认使用 mask 选取目标点。
- 影响:
  - 无法直接复用“基于外部分割 mask 的输入管线”。
- 解决方案:
  - 在 ROS 节点中增加 fallback 逻辑：使用 `depth > 0` 生成有效区域 mask，保证节点可运行。
  - 同时保留可扩展空间，后续可接入专用分割 topic 以提高精度与鲁棒性。

## Issue 3: 原模型实现中 `DataParallel` 不适配 CPU-only ROS 部署

- 问题现象:
  - 原始 DenseFusion 网络结构使用 `nn.DataParallel` 封装，在 CPU-only 部署时容易出现设备语义冲突。
- 影响:
  - ROS 节点在仅 CPU 环境下可能报错或行为不稳定。
- 解决方案:
  - 在 ROS 包内实现了 CPU-safe 的网络版本（去除 `DataParallel` 依赖），并统一 `map_location=\"cpu\"` 加载权重。

## Issue 4: ROS2 包需要“独立编译运行”

- 问题现象:
  - 直接依赖仓库根目录 `lib/`、`datasets/` 会导致包不可独立部署。
- 影响:
  - 违反“ROS package 单独编译和运行”的目标。
- 解决方案:
  - 将推理核心依赖（网络结构、几何变换、预处理逻辑）打包到 `densefusion_ros/densefusion_ros/densefusion_core`。
  - 通过 `ament_python` + `setup.py` 提供独立安装入口与启动脚本。

## 验证结果

- 构建命令:
  - `colcon build --packages-select densefusion_ros --base-paths densefusion_ros`
- 构建结果:
  - `1 package finished`，构建成功。

## Issue 5: ROS 节点需要去除 Torch 依赖并改为 ONNX CPU 推理

- 问题现象:
  - 旧版节点通过 PyTorch 模型推理，运行时需要 `torch/torchvision`。
- 影响:
  - 不符合“ROS 节点端仅 ONNX CPU 推理、移除 torch 依赖”的目标。
- 解决方案:
  - 将节点推理链路切换到 `onnxruntime`（`CPUExecutionProvider`）。
  - 新增 `OnnxDenseFusion` 推理器，分别加载 `pose_onnx_path` 与 `refine_onnx_path`。
  - `setup.py` 中移除 `torch/torchvision`，仅保留 `onnxruntime` 等必要依赖。
  - launch 参数同步切换为 ONNX 模型路径参数。

## Issue 6: ROS2 运行时 Python 与 conda Python 环境不一致

- 问题现象:
  - `ros2 launch` 启动节点时报错 `ModuleNotFoundError: No module named 'onnxruntime'`。
- 原因:
  - 节点运行时使用系统 Python（ROS2 Jazzy），而 `onnxruntime` 仅安装在 conda 环境。
- 解决方案:
  - 为系统 Python 安装 `onnxruntime`：
  - `/usr/bin/python3 -m pip install --user --break-system-packages onnxruntime`

## Issue 7: 将 conda site-packages 注入 ROS2 进程会触发 NumPy ABI 冲突

- 问题现象:
  - 通过 `PYTHONPATH` 注入 conda 包后，`cv_bridge` 报错：
  - `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
- 原因:
  - ROS2 `cv_bridge` 二进制扩展与系统 NumPy ABI 绑定，和 conda 的 NumPy 版本不兼容。
- 解决方案:
  - 不使用 conda `PYTHONPATH` 覆盖 ROS2 运行时环境。
  - 使用系统 ROS 环境运行：
  - `source /opt/ros/jazzy/setup.bash && source install/setup.bash`

## Issue 8: Orbbec D335 话题名称与原默认配置不一致

- 问题现象:
  - 相机实际发布 `/camera/color/image_raw` 与 `/camera/depth/image_raw`。
  - 旧默认值为 `/camera/color/raw_image` 与 `/camera/depth/image`。
- 解决方案:
  - 已更新 ROS 节点、launch、本地模拟发布器和 README 默认话题为 D335 话题。

## 本地文件模拟验证（Linemod_preprocessed）

- 测试输入:
  - RGB: `datasets/Linemod_preprocessed/Linemod_preprocessed/data/01/rgb/0000.png`
  - Depth: `datasets/Linemod_preprocessed/Linemod_preprocessed/data/01/depth/0000.png`
- 启动方式:
  - `ros2 launch densefusion_ros local_file_test.launch.py ...`
- 结果:
  - `file_input_publisher` 持续发布到 `/camera/color/image_raw`、`/camera/depth/image_raw`
  - `densefusion_ros_node` 成功持续发布 `/pose_stamp`（日志显示多帧发布成功）

## Issue 9: FastSAM ONNX 外部权重文件命名不匹配

- 问题现象:
  - 直接加载 `fast-sam-x.onnx` 报错缺少 `model.data`，而目录中实际文件名是 `fast-sam-x.data`。
- 原因:
  - ONNX 外部数据引用的是默认名 `model.data`。
- 解决方案:
  - 在 FastSAM 推理器初始化时自动检查并创建 `model.data -> fast-sam-x.data` 软链接。

## Issue 10: FastSAM mask 在 LineMOD 上与目标实例 mask 差异较大

- 问题现象:
  - 使用 `fast-sam-x.onnx` 对 `Linemod_preprocessed` 跑 50 张样本 benchmark：
  - GT vs FastSAM mask 平均 IoU 约 `0.0000`，位姿平移平均差约 `135.374 mm`。
- 影响:
  - 当前 FastSAM 直接“取最高分实例”策略不能稳定对齐 LineMOD 指定目标，导致 mask 质量偏低。
- 解决方案:
  - 作为功能验证链路，ROS 节点已可基于 FastSAM mask 稳定发布 pose。
  - 后续建议接入目标选择策略（prompt/ROI/跟踪先验）或与检测框结合筛选实例，再用于 DenseFusion。
