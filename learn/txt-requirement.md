# Orin 后续改进方向指导文档

本文档用于记录项目迁移到 Orin 之后的后续改进方向。它不是当前部署教程，也不是必须一次性完成的任务清单。

当前阶段的原则是：先让现有代码在 Orin 上跑通，再逐步优化摄像头输入、硬件解码、服务化、性能和模型工程。

## 一、当前代码已经具备的能力

现有项目已经具备以下核心能力：

- 使用 `TensorRT engine` 做本体检测，检测器位于 `pipeline/detector.py`。
- 使用 `PyCUDA` 管理图形处理器内存和推理缓冲。
- 使用本地 `FAISS` 向量库做语义检索。
- 使用远程 `Embedding` 服务生成文本向量。
- 使用远程 `VLM` 服务识别船号和船体描述。
- 支持视频文件输入。
- 支持普通 USB 摄像头输入，例如 `0`、`1`。
- 支持 `RTSP/HTTP` 网络视频流输入。
- 支持网页侧启动视频任务和摄像头任务。
- 支持后端推理后通过 `H.264 + WebSocket` 推送结果画面到网页。

当前推荐先按下面的输入源验收：

```text
USB 摄像头：0
RTSP 网络摄像头：rtsp://摄像头地址/stream
视频文件：mp4、avi、mkv 等
```

## 二、当前不优先修改的内容

下面这些模块当前不作为第一批改造目标：

- 不改数据库逻辑，继续使用当前 `CSV/SQLite` 双后端结构。
- 不改向量库逻辑，继续保持盒子本地 `FAISS`，远程只生成向量。
- 不改数据管理 Demo。
- 不改视频 Demo 的主要交互流程。
- 不把 `FAISS` 迁移到远程服务器。
- 不把 `VLM` 或 `Embedding` 模型放到 Orin 本地运行。
- 不重构整体 Web 架构。

这样做的目标是降低迁移风险：先让本体检测、远程识别、本地检索和网页播放完整跑通。

## 三、后续改进一：MIPI CSI 摄像头适配

目标：

支持 Orin 排线直连摄像头，例如 `IMX219`、`IMX477`，并允许使用类似下面的输入：

```bash
python -m pipeline mipi://0 --camera
```

为什么需要：

当前 `pipeline/video_input.py` 主要通过 `cv2.VideoCapture()` 打开数字编号、视频文件和网络流。USB 摄像头和 RTSP 可以先用，但 MIPI CSI 摄像头在 Jetson 平台上通常需要走 `GStreamer` 管线，常见入口是 `nvarguscamerasrc`。

涉及模块：

```text
pipeline/video_input.py
web/static/js/pipeline.js
web/routes/pipeline_api.py
config.yaml
```

建议实现方向：

- 在 `InputSource` 中识别 `mipi://0`、`mipi://1` 这类输入。
- 将 `mipi://0` 转换为 `nvarguscamerasrc sensor-id=0` 管线。
- 通过 `cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)` 打开。
- 配置宽高、帧率、格式，默认可以先用 `1280x720@30`。
- 在网页摄像头 Demo 中增加 MIPI 输入选项。

示例管线：

```text
nvarguscamerasrc sensor-id=0 !
video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 !
nvvidconv !
video/x-raw, format=BGRx !
videoconvert !
video/x-raw, format=BGR !
appsink drop=true max-buffers=1
```

是否必须：

仅当实际使用 MIPI CSI 摄像头时必须做。如果使用 USB 摄像头或 RTSP 网络摄像头，可以暂时不做。

验收标准：

- `mipi://0` 能打开并读到帧。
- 连续运行十分钟不明显卡死。
- 摄像头 Demo 能显示推理后画面。
- 日志能输出实际分辨率和帧率。

## 四、后续改进二：RTSP 硬件解码

目标：

让 RTSP 网络流优先使用 Orin 的硬件解码能力，降低中央处理器占用。

为什么需要：

当前 RTSP 可以直接交给 `cv2.VideoCapture(rtsp_url)`，这能跑通功能，但是否使用硬件解码取决于系统 OpenCV 和后端配置。高分辨率视频流或多路流场景下，软件解码会明显占用中央处理器。

涉及模块：

```text
pipeline/video_input.py
config.yaml
```

建议实现方向：

- 为 RTSP 输入增加可配置解码模式：`auto`、`opencv`、`gstreamer_hw`。
- 默认保留当前 `opencv` 或 `auto`，避免影响已有行为。
- 当启用 `gstreamer_hw` 时，使用 `rtspsrc + nvv4l2decoder + nvvidconv`。
- 增加失败回退：硬件管线打不开时，回退到当前 `cv2.VideoCapture(rtsp_url)`。

示例管线：

```text
rtspsrc location=rtsp://摄像头地址/stream latency=100 !
rtph264depay !
h264parse !
nvv4l2decoder !
nvvidconv !
video/x-raw, format=BGRx !
videoconvert !
video/x-raw, format=BGR !
appsink drop=true max-buffers=1
```

是否必须：

单路 RTSP 演示不是必须。多路摄像头、长时间运行、低功耗部署时建议做。

验收标准：

- RTSP 输入能正常打开。
- 推理画面延迟稳定。
- 中央处理器占用低于软件解码方案。
- 硬件管线失败时能自动回退，并在日志中说明。

## 五、后续改进三：系统服务化

目标：

让项目作为 Orin 上的系统服务运行，支持开机自启、崩溃重启和日志管理。

为什么需要：

演示阶段可以手动执行 `python -m web`，但现场部署需要服务自动恢复，不能依赖人工打开终端。

涉及模块：

```text
deploy/ship-service.service
deploy/install.sh
config.yaml
```

建议实现方向：

- 新增 `systemd` 服务文件。
- 固定项目部署目录，例如 `/opt/ship-detection`。
- 使用虚拟环境中的 Python 启动 Web 服务。
- 配置 `Restart=always` 和 `RestartSec=5`。
- 日志统一用 `journalctl` 查看。

示例服务内容：

```ini
[Unit]
Description=Ship Detection Web Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/ship-detection
ExecStart=/opt/ship-detection/.venv/bin/python -m web
Restart=always
RestartSec=5
Environment=PYTHONPATH=/opt/ship-detection
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

是否必须：

现场长期部署时必须。开发调试阶段可以先不做。

验收标准：

- 开机后服务自动启动。
- 访问 Web 页面正常。
- 进程异常退出后可以自动重启。
- `journalctl -u ship-detection -f` 能看到日志。

## 六、后续改进四：Orin 性能优化

目标：

在保证识别效果的前提下，降低延迟、稳定帧率、控制内存和功耗。

为什么需要：

Orin 算力强，但视频流、检测、远程识别、网页推流叠加后，仍然需要控制队列积压和资源占用。

涉及模块：

```text
pipeline/pipeline.py
pipeline/video_input.py
web/routes/pipeline_api.py
config.yaml
```

建议优化项：

- 设置合理的 `target_fps`，避免无意义满速读取视频流。
- 调整 `detect_every_n_frames`，减少检测频率。
- 调整 `process_every_n_frames`，减少远程 VLM 请求频率。
- 控制 `max_queued_frames`，避免识别任务堆积。
- 增加内存监控，内存过高时降低处理频率。
- 增加功耗模式说明，部署时使用合适的 `nvpmodel`。

推荐先从配置层调整：

```yaml
pipeline:
  target_fps: 15
  detect_every_n_frames: 2
  process_every_n_frames: 15
  max_queued_frames: 30
  max_concurrent: 4
```

是否必须：

基础演示不是必须。现场稳定运行建议做。

验收标准：

- 连续运行一小时不卡死。
- Web 画面延迟不持续增长。
- 任务队列不会长期满载。
- 内存占用保持稳定。

## 七、后续改进五：模型工程化

目标：

建立稳定的模型导出、版本管理和验收流程。

为什么需要：

`.engine` 文件和 Orin 硬件、CUDA、TensorRT 版本强相关。直接复用其他机器生成的 `.engine` 可能加载失败或输出格式不一致。

涉及模块：

```text
models/
tools/
pipeline/detector.py
config.yaml
```

建议实现方向：

- 在 Orin 本机或同版本环境生成 `.engine`。
- 将模型文件放入固定目录，例如 `models/yolov8_ship_fp16.engine`。
- 在配置中写清楚模型路径。
- 记录模型来源、类别编号、输入尺寸、导出时间和 TensorRT 版本。
- 后续支持 `FP16` 导出。
- 如果需要更高性能，再规划 `INT8` 量化和校准数据集。

是否必须：

`.engine` 本机可用是必须。完整模型工程化流程可以后续补。

验收标准：

- `ShipDetector` 能加载 `.engine`。
- 单张测试图能输出合理检测框。
- 输出类别编号和当前 `detect_classes` 匹配。
- 更换模型后能通过配置切换。

## 八、后续改进优先级

第一优先级：

- Orin 上跑通当前代码。
- 确认 `.engine` 能加载。
- 确认远程 `Embedding/VLM` 可用。
- 确认 USB 摄像头或 RTSP 至少一种输入可用。

第二优先级：

- RTSP 硬件解码。
- 系统服务化。
- 性能参数调优。

第三优先级：

- MIPI CSI 摄像头适配。
- 模型导出工具化。
- 功耗和内存自动监控。

## 九、明确边界

本文档只作为后续开发路线图，不代表当前必须全部实现。

当前部署教程请看：

```text
learn/requirement.md
```

历史迁移需求笔记保留在：

```text
learn/reqirment.md
```
