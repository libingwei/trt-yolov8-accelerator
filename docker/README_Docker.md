# Docker开发环境使用指南

## 🐳 快速开始

### 1. 构建Docker镜像

```bash
cd /Users/koo/Code/diy/AI-Infer-Acc/projects/trt-yolov8-accelerator/docker
chmod +x docker-dev.sh

# 构建镜像（包含TensorRT、CUDA开发工具、SSH服务）
./docker-dev.sh build
```

### 2. 运行容器

```bash
# 运行容器并设置自定义密码
./docker-dev.sh run -p your_secure_password

# 或使用默认密码
./docker-dev.sh run
```

### 3. 连接方式

#### 方式1: SSH连接（推荐）
```bash
# 查看连接信息
./docker-dev.sh ssh-info

# 通过端口映射连接
ssh root@localhost -p 2222

# 通过容器IP连接（需要先获取IP）
docker inspect trt-yolov8-container | grep IPAddress
ssh root@<container_ip>
```

#### 方式2: 直接进入容器
```bash
./docker-dev.sh shell
```

## 🔧 开发环境验证

连接到容器后，验证环境：

```bash
# 验证CUDA
nvcc --version
nvidia-smi

# 验证TensorRT
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"

# 验证OpenCV
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# 编译测试项目
cd /workspace
mkdir build && cd build
cmake .. -DYOLO_BUILD_PLUGINS=OFF
make -j$(nproc)

# 测试可执行文件
ls -la bin/
./bin/onnx_to_trt_yolo --help
```

## 📋 可用命令

```bash
./docker-dev.sh build              # 构建镜像
./docker-dev.sh run -p password    # 运行容器
./docker-dev.sh start              # 启动已存在容器
./docker-dev.sh stop               # 停止容器
./docker-dev.sh shell              # 进入容器shell
./docker-dev.sh logs               # 查看容器日志
./docker-dev.sh ssh-info           # 显示SSH连接信息
./docker-dev.sh clean              # 清理容器和镜像
```

## 🔐 安全设置

### 修改默认密码
```bash
# 运行时设置
./docker-dev.sh run -p your_secure_password

# 或在容器内修改
passwd root
```

### SSH密钥认证（可选）
```bash
# 在容器内设置
mkdir -p /root/.ssh
# 将你的公钥添加到 /root/.ssh/authorized_keys
echo "your_public_key_here" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
chmod 700 /root/.ssh
```

## 🔍 故障排除

### 1. 容器无法启动
```bash
# 查看详细日志
./docker-dev.sh logs

# 检查Docker和NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.4-runtime nvidia-smi
```

### 2. SSH连接被拒绝
```bash
# 检查SSH服务状态
./docker-dev.sh shell
systemctl status ssh

# 重启SSH服务
service ssh restart
```

### 3. nvcc命令未找到
```bash
# 检查环境变量
echo $CUDA_HOME
echo $PATH

# 重新加载环境
source /etc/environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### 4. 权限问题
```bash
# 确保Docker有GPU访问权限
docker run --rm --gpus all nvidia/cuda:12.4-runtime nvidia-smi

# 检查用户组
groups $USER
# 如果不在docker组，添加用户到docker组
sudo usermod -aG docker $USER
```

## 🚀 开发工作流

### YOLOv8模型转换
```bash
# 1. 进入容器
./docker-dev.sh shell

# 2. 导出ONNX模型
cd /workspace/projects/trt-yolov8-accelerator
python scripts/export_yolov8_onnx.py --weights yolov8n.pt --outdir models

# 3. 构建项目
cd /workspace && mkdir build && cd build
cmake .. -DYOLO_BUILD_PLUGINS=OFF
make -j$(nproc)

# 4. 转换为TensorRT引擎
./bin/onnx_to_trt_yolo models/yolov8n.onnx models/yolov8n.trt --fp16

# 5. 测试推理
./bin/yolo_trt_infer models/yolov8n.trt --image assets/sample.jpg
```

### 视频追踪开发
```bash
# 构建视频追踪模块
cd /workspace/build
make video_tracking_lib

# 运行测试
./bin/tracking_test
./bin/advanced_test
```

## 📊 性能监控

### 容器资源使用
```bash
# 监控容器资源
docker stats trt-yolov8-container

# GPU使用情况
nvidia-smi -l 1
```

### 开发环境基准测试
```bash
# 在容器内运行
cd /workspace/projects/trt-yolov8-accelerator/video_tracking
./build_standalone/bin/advanced_test  # 性能测试模式
```

## 🔄 数据持久化

项目目录通过Docker volume挂载，所有代码修改会自动同步：
- 主机目录: `/Users/koo/Code/diy/AI-Infer-Acc`
- 容器目录: `/workspace`

生成的模型文件、编译结果都会保存在主机上。
