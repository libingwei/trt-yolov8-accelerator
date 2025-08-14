# NVIDIA TensorRT 容器版本对应表

## 🔍 如何确定合适的TensorRT容器版本

### 📋 TensorRT容器版本 vs CUDA版本对应表

根据NVIDIA官方文档整理的主要版本对应关系：

| TensorRT容器标签 | TensorRT版本 | CUDA版本 | Ubuntu版本 | Python版本 | 推荐用途 |
|----------------|-------------|----------|-----------|-----------|----------|
| `24.08-py3` | 10.3.0 | 12.6 | 22.04 | 3.10 | 最新版本 |
| `24.07-py3` | 10.2.0 | 12.5 | 22.04 | 3.10 | 稳定版本 |
| `24.06-py3` | 10.1.0 | 12.5 | 22.04 | 3.10 | - |
| `24.05-py3` | 10.0.1 | 12.4 | 22.04 | 3.10 | - |
| `24.04-py3` | 10.0.0 | 12.4 | 22.04 | 3.10 | - |
| `24.03-py3` | 9.3.0 | 12.4 | 22.04 | 3.10 | **当前使用** |
| `24.02-py3` | 9.2.0 | 12.3 | 22.04 | 3.10 | - |
| `24.01-py3` | 9.1.0 | 12.3 | 22.04 | 3.10 | - |
| `23.12-py3` | 8.6.1 | 12.3 | 22.04 | 3.10 | - |
| `23.11-py3` | 8.6.1 | 12.2 | 22.04 | 3.10 | - |
| `23.10-py3` | 8.6.1 | 12.2 | 22.04 | 3.10 | - |

### 🎯 根据主机CUDA版本选择容器

**你的情况：主机支持CUDA 12.7**

推荐选择：
1. **首选**: `24.08-py3` (TensorRT 10.3.0, CUDA 12.6) - 最接近你的CUDA版本
2. **次选**: `24.07-py3` (TensorRT 10.2.0, CUDA 12.5) - 稳定版本
3. **保守**: `24.05-py3` (TensorRT 10.0.1, CUDA 12.4) - 如果需要更稳定

### 📝 验证容器版本信息的方法

#### 方法1: 直接运行容器查看版本
```bash
# 查看TensorRT版本
docker run --rm nvcr.io/nvidia/tensorrt:24.08-py3 python3 -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"

# 查看CUDA版本
docker run --rm nvcr.io/nvidia/tensorrt:24.08-py3 nvcc --version

# 查看完整系统信息
docker run --rm nvcr.io/nvidia/tensorrt:24.08-py3 cat /etc/os-release
```

#### 方法2: 检查容器标签和文档
```bash
# 拉取并检查镜像信息
docker pull nvcr.io/nvidia/tensorrt:24.08-py3
docker inspect nvcr.io/nvidia/tensorrt:24.08-py3 | grep -A 5 -B 5 "CUDA\|TensorRT"
```

#### 方法3: 访问NVIDIA NGC官方页面
访问: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt

### 🔧 兼容性规则

1. **CUDA向下兼容**: 主机CUDA 12.7可以运行CUDA 12.6及以下的容器
2. **TensorRT版本**: 新版本通常向下兼容
3. **Driver要求**: 确保NVIDIA驱动支持容器中的CUDA版本

### 🚀 推荐的Dockerfile修改

基于你的CUDA 12.7环境，推荐使用：

```dockerfile
# 选择1: 最新稳定版 (推荐)
FROM nvcr.io/nvidia/tensorrt:24.08-py3

# 选择2: 当前稳定版
FROM nvcr.io/nvidia/tensorrt:24.07-py3

# 选择3: 保守稳定版 (如果遇到兼容性问题)
FROM nvcr.io/nvidia/tensorrt:24.05-py3
```

### ⚠️ 注意事项

1. **开发工具包**: TensorRT容器已包含运行时，但可能需要额外安装开发工具
2. **cuDNN版本**: 通常与CUDA版本匹配，容器已预装
3. **Python包**: 容器已预装python3-libnvinfer等必要包

### 🔍 实际测试建议

```bash
# 1. 测试容器是否能正常运行
docker run --rm --gpus all nvcr.io/nvidia/tensorrt:24.08-py3 nvidia-smi

# 2. 测试TensorRT功能
docker run --rm --gpus all nvcr.io/nvidia/tensorrt:24.08-py3 python3 -c "
import tensorrt as trt
print(f'TensorRT version: {trt.__version__}')
print('TensorRT working correctly!')
"

# 3. 测试CUDA编译
docker run --rm --gpus all nvcr.io/nvidia/tensorrt:24.08-py3 sh -c "
apt-get update && apt-get install -y cuda-compiler-12-6 cuda-nvcc-12-6
nvcc --version
"
```

### 📊 性能对比参考

| 版本 | TensorRT性能 | 兼容性 | 稳定性 | 推荐度 |
|------|-------------|--------|--------|--------|
| 24.08-py3 | 最高 | 良好 | 新版本 | ⭐⭐⭐⭐⭐ |
| 24.07-py3 | 高 | 很好 | 稳定 | ⭐⭐⭐⭐⭐ |
| 24.05-py3 | 高 | 很好 | 很稳定 | ⭐⭐⭐⭐ |
| 24.03-py3 | 中高 | 很好 | 很稳定 | ⭐⭐⭐ |

**结论**: 建议从`24.07-py3`开始尝试，如果需要最新特性则选择`24.08-py3`。
