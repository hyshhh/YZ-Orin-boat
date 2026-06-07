# 当前代码 Orin 环境配置与验收教程

本文档用于在不修改当前代码的前提下，把项目部署到 Orin 64G + Ubuntu 20.04 环境，并完成基础验收。

本次核对后的结论很明确：当前部署方案可行，但 Python 3.10 与 JetPack 5.x 自带 TensorRT 绑定之间存在兼容风险，不能把它写成普通 `pip install` 就一定能解决的依赖。

当前部署方案：

```text
Orin 盒子
├── TensorRT 本体检测
├── 本地 SQLite/CSV 数据库
├── 本地 FAISS 向量库
├── 本地 Web 服务
└── 远程调用 Embedding 与 VLM 服务

远程服务器
├── Embedding 服务
└── VLM 服务
```

## 一、准确性核对结论

根据英伟达和包索引资料，本文档按下面结论修正：

- JetPack 5.1.5 对应 Jetson Linux 35.6.x，根文件系统基于 Ubuntu 20.04，包含 CUDA 11.4.19、TensorRT 8.5.2、cuDNN 8.6.0 和 OpenCV 4.5.4。
- Ubuntu 20.04 默认 Python 主版本是 3.8，不是 3.10；项目代码要求 Python 3.10 以上，因此 Python 3.10 需要作为单独运行环境处理。
- JetPack 5.x 的 TensorRT Python 包通过系统包安装，常见绑定是系统 Python 路径；Python 3.10 虚拟环境能否直接导入 `tensorrt` 必须现场验证。
- `faiss-cpu` 当前在 PyPI 上支持 Python 3.10 以上，并提供 Linux ARM64 轮子；盒子本地安装 `faiss-cpu` 是合理的。
- `langchain_community.vectorstores.FAISS` 只是上层封装，仍然需要底层 `faiss-cpu`。
- `opencv-python` 是预构建 CPU 包，不等同于 JetPack 自带的硬件加速 OpenCV；当前代码普通视频、USB 摄像头、RTSP 可先验收，MIPI CSI 和硬件解码不在本教程范围内。
- `ultralytics` 会牵涉 PyTorch。Jetson 上 PyTorch 应优先按英伟达 Jetson 专用轮子安装；如果只加载 `.engine` 推理，`ultralytics` 不是基础运行的必需项。
- TensorRT `.engine` 默认不建议跨平台复用，最好在 Orin 本机或同架构、同 JetPack、同 TensorRT 版本环境生成。

## 二、适用范围

本教程适用于：

- Orin 64G。
- Ubuntu 20.04。
- JetPack 5.x，优先建议 5.1.x。
- 当前代码不做功能修改。
- 摄像头先使用 USB 摄像头或 RTSP 网络摄像头。
- 远程服务器已经提供 `Embedding` 和 `VLM` 服务。

本教程不包含：

- MIPI CSI 摄像头适配。
- RTSP 硬件解码改造。
- `systemd` 服务化部署。
- 当前代码语法降级到 Python 3.8。
- 本地部署大模型。

## 三、部署前确认

在 Orin 上确认系统信息：

```bash
uname -m
cat /etc/os-release
```

期望结果：

```text
架构：aarch64
系统：Ubuntu 20.04
```

确认 JetPack 相关组件：

```bash
nvcc --version
dpkg -l | grep nvinfer
dpkg -l | grep cudnn
dpkg -l | grep nvidia-jetpack
```

如果 JetPack 组件缺失，优先使用英伟达的系统包方式补齐：

```bash
sudo apt update
sudo apt install -y nvidia-jetpack
```

确认远程服务器可访问：

```bash
ping 远程服务器地址
```

如果远程服务器禁用 `ping`，可以直接测试端口：

```bash
curl http://远程服务器地址:7891/v1/models
curl http://远程服务器地址:7890/v1/models
```

如果远程服务没有 `/v1/models` 接口，后面直接测试 `/v1/embeddings` 和 `/v1/chat/completions`。

## 四、安装系统依赖

更新系统包：

```bash
sudo apt update
sudo apt upgrade -y
```

安装基础工具：

```bash
sudo apt install -y \
  build-essential \
  cmake \
  pkg-config \
  git \
  curl \
  wget \
  ffmpeg
```

安装视频相关依赖：

```bash
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly
```

安装数值计算依赖：

```bash
sudo apt install -y \
  libopenblas-dev \
  libomp-dev
```

安装系统 Python 工具：

```bash
sudo apt install -y \
  python3-pip \
  python3-venv \
  python3-dev
```

说明：

- Ubuntu 20.04 默认 `python3` 通常是 Python 3.8。
- 不要替换系统默认 `python3`，否则可能影响 JetPack 和系统工具。
- 当前项目代码要求 Python 3.10 以上，下一节单独处理。

## 五、Python 运行环境选择

当前项目的 `pyproject.toml` 写明 `requires-python = ">=3.10"`，代码中也使用了 Python 3.10 语法。因此“不改代码直接运行”的路线必须准备 Python 3.10 环境。

但是这里有一个关键风险：JetPack 5.x 自带的 `TensorRT` Python 绑定通常跟系统 Python 绑定，Python 3.10 虚拟环境能否导入 `tensorrt` 不能提前保证。

推荐先尝试 Python 3.10 直迁路线：

```bash
python3.10 --version
```

如果命令不存在，说明当前系统源没有直接提供 Python 3.10，需要二选一：

- 使用可信额外源或源码方式安装 Python 3.10。
- 暂停直迁，改做 Python 3.8 兼容改造；这会涉及业务代码修改，不属于当前教程。

如果已经有 Python 3.10，创建虚拟环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

不要在没有验证 TensorRT 的情况下就认定环境完成。必须继续执行第八节的导入检查。

## 六、准备项目目录

建议把项目放到固定目录：

```bash
sudo mkdir -p /opt/ship-detection
sudo chown -R $USER:$USER /opt/ship-detection
```

进入项目目录：

```bash
cd /opt/ship-detection
```

如果项目已经在其他目录，也可以直接在实际项目目录操作，不强制使用 `/opt/ship-detection`。

## 七、安装项目依赖

基础依赖：

```bash
pip install \
  fastapi \
  "uvicorn[standard]" \
  jinja2 \
  python-multipart \
  pyyaml \
  langchain-core \
  langchain-community \
  langchain-openai \
  httpx \
  numpy \
  faiss-cpu \
  rich
```

视频依赖先按下面方式安装：

```bash
pip install opencv-python
```

说明：

- `faiss-cpu` 是必须的，`langchain_community.vectorstores.FAISS` 只是上层封装。
- 盒子本地保存 FAISS 索引，远程服务器只负责生成向量，不负责检索。
- `opencv-python` 可用于当前代码的基础视频处理验收，但不承诺 GStreamer、MIPI CSI 或硬件解码能力。
- JetPack 自带 OpenCV 版本和 `pip install opencv-python` 版本可能不同，优先以“当前代码能否打开视频源”为验收标准。

可选依赖：

```bash
pip install ultralytics
```

只有在需要导出模型、转换模型或使用 Ultralytics 工具链时才安装它。它可能拉取 `torch`，Jetson 上不建议随意安装普通 PyPI 的 GPU 版 PyTorch，应优先按英伟达 Jetson 专用安装方式处理。

完整项目依赖以 `pyproject.toml` 为准，其中包含 `aiortc`。当前基础验收不强制安装 `aiortc`；如果后续网页摄像头链路明确报缺失，再按报错补装：

```bash
pip install aiortc
```

## 八、确认 TensorRT 和 PyCUDA

优先使用 JetPack 自带的 TensorRT 包，不建议盲目从普通 PyPI 源安装 `tensorrt`。

先检查系统 Python 是否能导入 TensorRT：

```bash
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

再检查当前 Python 3.10 虚拟环境是否能导入：

```bash
source .venv/bin/activate
python -c "import tensorrt as trt; print(trt.__version__)"
```

检查 PyCUDA：

```bash
python -c "import pycuda.driver as cuda; print('pycuda ok')"
```

如果系统 Python 可以导入 `tensorrt`，但 Python 3.10 虚拟环境不能导入，说明问题不是项目代码，而是 JetPack 的 Python 绑定路径或版本不兼容。此时常见处理方向：

- 继续处理 Python 3.10 环境下的 TensorRT 绑定路径。
- 在同一台 Orin 上使用容器或自建环境统一 Python 3.10、TensorRT、PyCUDA。
- 修改项目代码兼容 Python 3.8，再使用 JetPack 系统 Python；这不属于“当前代码不修改”的路线。

只有下面命令都通过，才算本机推理依赖通过：

```bash
python -c "import cv2; print(cv2.__version__)"
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "import pycuda.driver as cuda; print('pycuda ok')"
```

## 九、修改配置文件

只需要修改 `config.yaml` 中远程服务地址、模型名、密钥和检测模型路径。

远程 VLM 配置：

```yaml
llm:
  model: "Qwen/Qwen3-VL-4B-AWQ"
  api_key: "abc123"
  base_url: "http://远程服务器地址:7890/v1"
  temperature: 0.0
```

远程 Embedding 配置：

```yaml
embed:
  model: "Qwen3-Embedding-0.6B"
  api_key: "abc123"
  base_url: "http://远程服务器地址:7891/v1"
```

本地向量库配置：

```yaml
vector_store:
  persist_path: "./vector_store"
  auto_rebuild: false
```

检测模型配置：

```yaml
pipeline:
  yolo_model: "models/yolov8_ship.engine"
  device: "0"
  conf_threshold: 0.25
  detect_every_n_frames: 2
  process_every_n_frames: 15
```

说明：

- `Embedding` 和 `VLM` 在远程服务器。
- `FAISS` 索引保存在 Orin 本地。
- `.engine` 文件保存在 Orin 本地。

## 十、向量库验收

当前项目逻辑是：

```text
查询文本
-> 请求远程 Embedding 服务生成向量
-> 在 Orin 本地 FAISS 检索
-> 返回船号候选
```

远程服务器只需要提供：

```text
/v1/embeddings
/v1/chat/completions
```

测试 Embedding 服务：

```bash
python - <<'PY'
import httpx

resp = httpx.post(
    "http://远程服务器地址:7891/v1/embeddings",
    headers={"Authorization": "Bearer abc123"},
    json={"model": "Qwen3-Embedding-0.6B", "input": ["白色船体，蓝色上层建筑"]},
    timeout=30,
)
print(resp.status_code)
print(resp.text[:300])
PY
```

返回内容中应包含：

```text
data
embedding
```

如果更换 Embedding 模型，需要删除旧索引后重新建库：

```bash
rm -rf vector_store
```

## 十一、VLM 服务验收

测试远程 VLM 服务：

```bash
python - <<'PY'
import httpx

resp = httpx.post(
    "http://远程服务器地址:7890/v1/chat/completions",
    headers={"Authorization": "Bearer abc123"},
    json={
        "model": "Qwen/Qwen3-VL-4B-AWQ",
        "messages": [{"role": "user", "content": "你好"}],
    },
    timeout=30,
)
print(resp.status_code)
print(resp.text[:300])
PY
```

如果接口返回成功，说明盒子到远程 VLM 的基础链路可用。

## 十二、检测模型验收

`.engine` 文件建议在 Orin 本机，或同架构、同 JetPack、同 TensorRT 版本环境生成。

不建议直接复用普通电脑上生成的 `.engine` 文件。

确认模型文件存在：

```bash
ls -lh models/
```

测试检测器能否加载：

```bash
python - <<'PY'
from pipeline.detector import ShipDetector

d = ShipDetector(model_path="models/yolov8_ship.engine", device="0")
print("detector ok")
d.cleanup()
PY
```

如果这里失败，优先检查：

- `.engine` 路径是否正确。
- TensorRT 是否能导入。
- `.engine` 是否在 Orin 或同版本环境生成。
- 输出格式是否和当前 `pipeline/detector.py` 解码逻辑匹配。

## 十三、启动 Web 服务

在项目目录执行：

```bash
source .venv/bin/activate
python -m web
```

如果端口是默认配置：

```text
http://盒子地址:8000
```

也可以查看接口文档：

```text
http://盒子地址:8000/docs
```

## 十四、功能验收顺序

第一步，基础依赖导入：

```bash
python -c "import fastapi, yaml, httpx, numpy, faiss; print('base ok')"
python -c "import cv2; print(cv2.__version__)"
python -c "import tensorrt as trt; print(trt.__version__)"
python -c "import pycuda.driver as cuda; print('pycuda ok')"
```

第二步，远程服务连通：

```text
Embedding 服务测试通过
VLM 服务测试通过
```

第三步，本地向量库：

```text
能生成 vector_store/index.faiss
能执行语义检索
```

第四步，检测模型：

```text
ShipDetector 能加载 .engine
单张图或短视频能输出检测结果
```

第五步，网页功能：

```text
数据管理页面可打开
视频 Demo 可上传并处理视频
摄像头 Demo 可启动 USB 摄像头或 RTSP
网页能播放推理后的 H.264 结果流
```

## 十五、摄像头验收

命令行 USB 摄像头：

```bash
python -m pipeline 0 --camera
```

命令行 RTSP 摄像头：

```bash
python -m pipeline rtsp://摄像头地址/stream --camera
```

说明：

- 上面命令是命令行链路验收，不完全等同于网页摄像头 Demo 的全链路验收。
- 当前代码使用 `cv2.VideoCapture` 打开输入源，未专门适配 `nvarguscamerasrc`。
- 当前代码不保证 MIPI CSI 摄像头可用。MIPI CSI 摄像头适配作为后续改进方向，见 `learn/txt-requirement.md`。

## 十六、常见问题

`faiss-cpu` 安装失败：

```bash
python --version
uname -m
pip --version
```

确认 Python 为 3.10 以上，架构为 `aarch64`，并升级 `pip setuptools wheel`。

`tensorrt` 导入失败：

先用系统 Python 测试。如果系统 Python 可以导入，而虚拟环境不能导入，说明绑定路径或版本不兼容。这个问题需要优先解决，否则当前 TensorRT 检测器无法运行。

`cv2` 能导入但摄像头打不开：

先确认摄像头设备是否存在：

```bash
ls /dev/video*
```

再确认 OpenCV 视频能力：

```bash
python - <<'PY'
import cv2
print(cv2.getBuildInformation())
PY
```

远程服务正常但语义检索异常：

删除旧 `vector_store` 后重新建库，并确认建库和查询使用同一个 Embedding 模型。

`.engine` 无法加载：

重新在 Orin 本机或同版本环境导出 `.engine`，不要直接使用其他平台生成的引擎文件。

## 十七、当前部署通过标准

满足下面条件即可认为当前代码在 Orin 上完成基础部署：

- Web 服务可以启动。
- 远程 `Embedding` 和 `VLM` 可以访问。
- 本地 `FAISS` 可以建库和检索。
- `TensorRT engine` 可以加载。
- 视频文件可以处理。
- USB 摄像头或 RTSP 至少一种输入可以跑通。
- 网页可以看到推理后的结果流。

## 十八、参考资料

- 英伟达 JetPack 5.1.5 页面：https://developer.nvidia.com/embedded/jetpack-sdk-515
- 英伟达 Jetson Linux 35.6.2 页面：https://developer.nvidia.com/embedded/jetson-linux-r3562
- 英伟达 JetPack 安装文档：https://docs.nvidia.com/jetson/jetpack/5.1/install-jetpack/index.html
- 英伟达 Jetson PyTorch 安装文档：https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- TensorRT 引擎兼容性说明：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html
- PyPI `faiss-cpu` 页面：https://pypi.org/project/faiss-cpu/
- PyPI `opencv-python` 页面：https://pypi.org/project/opencv-python/
- LangChain FAISS 文档：https://docs.langchain.com/oss/python/integrations/vectorstores/faiss/
