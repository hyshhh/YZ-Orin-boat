# Orin平台迁移需求文档

## 一、当前项目状态分析

### 已完成的Orin适配 ✅

| 模块 | 状态 | 说明 |
|------|------|------|
| TensorRT检测器 | ✅ 已完成 | `pipeline/detector.py` 已实现TensorRT引擎推理 |
| CUDA加速 | ✅ 已完成 | 使用PyCUDA进行GPU内存管理 |
| H264编码 | ✅ 已完成 | 使用ffmpeg进行硬件编码 |
| WebSocket推流 | ✅ 已完成 | 支持fMP4 over WebSocket |

### 需要适配的模块 ⚠️

| 模块 | 优先级 | 说明 |
|------|--------|------|
| 摄像头输入 | 高 | 需要支持MIPI摄像头 |
| 系统服务 | 高 | 需要systemd服务化 |
| 功耗管理 | 中 | Orin功耗限制 |
| 内存优化 | 中 | 8GB内存限制 |
| 显示输出 | 低 | 本地显示支持 |

---

## 二、详细迁移需求

### 1. 摄像头输入适配

#### 1.1 MIPI摄像头支持

**当前状态**: 仅支持USB摄像头和RTSP流

**需求**:
- 支持MIPI CSI摄像头（如IMX219、IMX477）
- 使用GStreamer或V4L2接口
- 支持硬件解码

**修改文件**:
```
pipeline/video_input.py
```

**修改内容**:
```python
# 添加MIPI摄像头支持
class InputSource:
    def _open(self) -> None:
        # 添加MIPI摄像头判断
        if isinstance(source, str) and source.startswith("mipi://"):
            # 使用GStreamer管道
            gst_pipeline = (
                "nvarguscamerasrc sensor-id=0 ! "
                "video/x-raw(memory:NVMM), width=1920, height=1080, "
                "format=NV12, framerate=30/1 ! "
                "nvvidconv ! video/x-raw, format=BGRx ! "
                "videoconvert ! video/x-raw, format=BGR"
            )
            self._cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
```

#### 1.2 硬件解码支持

**需求**:
- RTSP流使用硬件解码
- 减少CPU占用

**修改文件**:
```
pipeline/video_input.py
```

**修改内容**:
```python
# RTSP硬件解码
if source.startswith("rtsp://"):
    gst_pipeline = (
        f"rtspsrc location={source} latency=100 ! "
        "rtph264depay ! h264parse ! "
        "nvv4l2decoder ! nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR"
    )
    self._cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
```

---

### 2. 系统服务化

#### 2.1 Systemd服务配置

**需求**:
- 开机自启动
- 崩溃自动重启
- 日志管理

**新建文件**:
```
deploy/ship-service.service
```

**文件内容**:
```ini
[Unit]
Description=Ship Detection Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ship-detection
ExecStart=/usr/bin/python3 -m web.app
Restart=always
RestartSec=5
Environment=PYTHONPATH=/opt/ship-detection
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

#### 2.2 安装脚本

**新建文件**:
```
deploy/install.sh
```

**文件内容**:
```bash
#!/bin/bash
# 安装脚本

# 1. 复制项目文件
cp -r . /opt/ship-detection

# 2. 安装依赖
cd /opt/ship-detection
pip3 install -r requirements.txt

# 3. 安装系统服务
cp deploy/ship-service.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable ship-service
systemctl start ship-service

echo "安装完成!"
```

---

### 3. 功耗管理

#### 3.1 功耗模式配置

**需求**:
- 支持多种功耗模式
- 根据负载自动调整

**新建文件**:
```
config/orin_power.yaml
```

**文件内容**:
```yaml
# Orin功耗配置
power:
  # 功耗模式: 15W / 30W / MAXN
  mode: "30W"

  # GPU频率限制 (MHz)
  gpu_freq_min: 300
  gpu_freq_max: 1300

  # CPU频率限制 (MHz)
  cpu_freq_min: 510
  cpu_freq_max: 2200

  # 自动调频
  auto_scaling: true
```

#### 3.2 功耗控制模块

**新建文件**:
```
pipeline/power_manager.py
```

**文件内容**:
```python
"""Orin功耗管理"""

import subprocess
import logging

logger = logging.getLogger(__name__)

class PowerManager:
    def __init__(self, mode: str = "30W"):
        self._mode = mode
        self._apply_mode(mode)

    def _apply_mode(self, mode: str):
        """应用功耗模式"""
        modes = {
            "15W": 15,
            "30W": 30,
            "MAXN": 60
        }
        power = modes.get(mode, 30)
        try:
            subprocess.run(
                ["nvpmodel", "-m", str(power)],
                check=True
            )
            logger.info("功耗模式已设置: %s", mode)
        except Exception as e:
            logger.error("设置功耗模式失败: %s", e)

    def set_mode(self, mode: str):
        """动态切换功耗模式"""
        self._mode = mode
        self._apply_mode(mode)
```

---

### 4. 内存优化

#### 4.1 内存监控

**需求**:
- 监控内存使用
- 低内存时降级处理

**修改文件**:
```
pipeline/pipeline.py
```

**修改内容**:
```python
def _check_memory(self) -> bool:
    """检查内存使用，返回是否充足"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        if mem.percent > 85:
            logger.warning("内存使用率过高: %.1f%%", mem.percent)
            return False
        return True
    except ImportError:
        return True
```

#### 4.2 帧缓冲优化

**修改文件**:
```
pipeline/video_input.py
```

**修改内容**:
```python
# 减少缓冲区大小
self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 仅保留1帧
```

---

### 5. 显示输出

#### 5.1 本地显示支持

**需求**:
- 支持HDMI输出
- 支持Web预览

**修改文件**:
```
pipeline/demo.py
```

**修改内容**:
```python
class DemoRenderer:
    def render(self, frame, ...):
        # 检查是否有显示器
        if os.environ.get("DISPLAY"):
            cv2.imshow("Ship Detection", frame)
        # 同时推送到WebSocket
        self._push_to_web(frame)
```

#### 5.2 硬件加速显示

**需求**:
- 使用NVMM内存
- 零拷贝显示

**新建文件**:
```
pipeline/display.py
```

**文件内容**:
```python
"""Orin硬件加速显示"""

import cv2

class OrinDisplay:
    def __init__(self):
        self._pipeline = None

    def show(self, frame):
        """使用GStreamer硬件加速显示"""
        if self._pipeline is None:
            self._pipeline = cv2.VideoWriter(
                "fakesink sync=false",
                cv2.CAP_GSTREAMER,
                0, 30,
                (frame.shape[1], frame.shape[0]),
                True
            )
        self._pipeline.write(frame)
```

---

### 6. 模型优化

#### 6.1 TensorRT引擎优化

**需求**:
- 针对Orin优化引擎
- INT8量化支持

**新建文件**:
```
tools/optimize_engine.py
```

**文件内容**:
```python
"""TensorRT引擎优化工具"""

import tensorrt as trt

def optimize_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1
):
    """优化ONNX模型为TensorRT引擎"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # 解析ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 配置优化
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    # 构建引擎
    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)

    return engine
```

---

### 7. 部署配置

#### 7.1 项目结构

```
ship-detection/
├── config/
│   ├── config.yaml          # 主配置
│   └── orin_power.yaml      # Orin功耗配置
├── deploy/
│   ├── ship-service.service # systemd服务
│   ├── install.sh           # 安装脚本
│   └── Dockerfile           # Docker部署
├── models/
│   └── yolov8n.engine       # TensorRT引擎
├── pipeline/
│   ├── detector.py          # TensorRT检测器
│   ├── video_input.py       # 视频输入
│   ├── power_manager.py     # 功耗管理
│   └── display.py           # 显示输出
├── web/
│   ├── app.py               # Web服务
│   └── routes/              # API路由
├── requirements.txt         # Python依赖
└── README.md
```

#### 7.2 依赖清单

**requirements.txt**:
```
# 核心依赖
fastapi>=0.104.0
uvicorn>=0.24.0
websockets>=12.0
opencv-python>=4.8.0
numpy>=1.24.0

# Orin专用依赖
tensorrt>=8.5.0
pycuda>=2022.2
Jetson.GPIO  # 可选：GPIO控制

# 监控依赖
psutil>=5.9.0
```

---

## 三、迁移步骤

### 第一阶段：基础适配 (1-2天)

1. ✅ TensorRT检测器已实现
2. ⚠️ 添加MIPI摄像头支持
3. ⚠️ 添加硬件解码支持

### 第二阶段：系统集成 (2-3天)

1. ⚠️ 创建systemd服务
2. ⚠️ 编写安装脚本
3. ⚠️ 配置开机自启

### 第三阶段：优化调优 (2-3天)

1. ⚠️ 功耗管理配置
2. ⚠️ 内存优化
3. ⚠️ 性能测试

### 第四阶段：测试验收 (1-2天)

1. ⚠️ 功能测试
2. ⚠️ 稳定性测试
3. ⚠️ 性能测试

---

## 四、注意事项

### 1. 硬件要求

- **开发板**: NVIDIA Jetson Orin Nano/NX
- **内存**: 8GB LPDDR5
- **存储**: 64GB eMMC 或 SD卡
- **摄像头**: USB摄像头 或 MIPI摄像头
- **网络**: 以太网或WiFi

### 2. 软件环境

- **系统**: JetPack 5.x (L4T 35.x)
- **CUDA**: 11.4+
- **TensorRT**: 8.5+
- **OpenCV**: 4.5+ (with GStreamer)

### 3. 性能指标

| 指标 | 目标值 |
|------|--------|
| YOLO检测帧率 | ≥30 FPS |
| 端到端延迟 | <100ms |
| 内存占用 | <4GB |
| 功耗 | <30W |

### 4. 已知限制

- INT8量化需要校准数据集
- MIPI摄像头需要设备树配置
- 功耗模式切换需要root权限

---

## 五、代码修改清单

### 必须修改

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `pipeline/video_input.py` | 添加MIPI摄像头支持 | 高 |
| `pipeline/video_input.py` | 添加硬件解码支持 | 高 |
| `config.yaml` | 添加Orin专用配置 | 高 |

### 建议修改

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `pipeline/pipeline.py` | 添加内存监控 | 中 |
| `pipeline/demo.py` | 添加本地显示支持 | 中 |
| `web/app.py` | 添加健康检查端点 | 中 |

### 新建文件

| 文件 | 说明 | 优先级 |
|------|------|--------|
| `deploy/ship-service.service` | systemd服务 | 高 |
| `deploy/install.sh` | 安装脚本 | 高 |
| `pipeline/power_manager.py` | 功耗管理 | 中 |
| `tools/optimize_engine.py` | 引擎优化工具 | 低 |

---

## 六、测试用例

### 1. 摄像头测试

```bash
# USB摄像头
python -m pipeline 0 --camera

# MIPI摄像头
python -m pipeline mipi://0 --camera

# RTSP流
python -m pipeline rtsp://192.168.1.100/stream --camera
```

### 2. 性能测试

```bash
# 检测帧率
python -c "from pipeline.detector import ShipDetector; d = ShipDetector(); import cv2; frame = cv2.imread('test.jpg'); import time; t=time.time(); [d.detect(frame) for _ in range(100)]; print(f'{100/(time.time()-t):.1f} FPS')"
```

### 3. 稳定性测试

```bash
# 运行24小时
python -m pipeline 0 --camera --max-frames 0 &
sleep 86400
```
