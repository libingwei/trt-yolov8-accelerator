#!/bin/bash

# TensorRT容器版本验证脚本

set -e

echo "🔍 TensorRT容器版本验证工具"
echo "=============================="

# 默认测试的版本列表（针对CUDA 12.7主机）
VERSIONS=(
    "24.08-py3"  # TensorRT 10.3.0, CUDA 12.6 - 最新
    "24.07-py3"  # TensorRT 10.2.0, CUDA 12.5 - 推荐  
    "24.05-py3"  # TensorRT 10.0.1, CUDA 12.4 - 稳定
    "24.03-py3"  # TensorRT 9.3.0, CUDA 12.4 - 当前
)

# 函数：测试单个版本
test_version() {
    local version=$1
    echo ""
    echo "📋 测试版本: nvcr.io/nvidia/tensorrt:${version}"
    echo "----------------------------------------"
    
    # 检查镜像是否存在
    if ! docker pull nvcr.io/nvidia/tensorrt:${version} >/dev/null 2>&1; then
        echo "❌ 镜像不存在或无法拉取"
        return 1
    fi
    
    echo "✅ 镜像拉取成功"
    
    # 测试基本功能
    echo "🔧 测试基本功能..."
    
    # 测试CUDA
    if docker run --rm --gpus all nvcr.io/nvidia/tensorrt:${version} nvidia-smi >/dev/null 2>&1; then
        echo "✅ CUDA运行正常"
    else
        echo "❌ CUDA运行失败"
        return 1
    fi
    
    # 获取版本信息
    echo "📊 版本信息:"
    docker run --rm nvcr.io/nvidia/tensorrt:${version} bash -c "
        echo 'TensorRT版本:' \$(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || echo '未知')
        echo 'CUDA版本:' \$(nvcc --version 2>/dev/null | grep 'release' | awk '{print \$6}' | sed 's/V//' || echo '未检测到nvcc')
        echo 'Python版本:' \$(python3 --version)
        echo 'Ubuntu版本:' \$(cat /etc/os-release | grep VERSION_ID | cut -d'\"' -f2)
    "
    
    # 测试TensorRT导入
    echo "🐍 测试TensorRT Python导入..."
    if docker run --rm nvcr.io/nvidia/tensorrt:${version} python3 -c "import tensorrt; print('TensorRT导入成功')" >/dev/null 2>&1; then
        echo "✅ TensorRT Python导入成功"
    else
        echo "❌ TensorRT Python导入失败"
        return 1
    fi
    
    return 0
}

# 函数：详细测试单个版本
detailed_test() {
    local version=$1
    echo ""
    echo "🔬 详细测试版本: nvcr.io/nvidia/tensorrt:${version}"
    echo "============================================"
    
    # 测试编译环境
    echo "🛠️ 测试编译环境..."
    docker run --rm nvcr.io/nvidia/tensorrt:${version} bash -c "
        apt-get update >/dev/null 2>&1
        apt-get install -y build-essential cmake >/dev/null 2>&1
        echo '✅ 基础编译工具安装成功'
        
        # 测试简单C++编译
        echo '#include <iostream>
int main() { std::cout << \"C++ compiler working\" << std::endl; return 0; }' > test.cpp
        g++ test.cpp -o test && ./test && echo '✅ C++编译测试成功'
    " 2>/dev/null || echo "❌ 编译环境测试失败"
    
    # 测试OpenCV
    echo "📷 测试OpenCV..."
    docker run --rm nvcr.io/nvidia/tensorrt:${version} bash -c "
        apt-get update >/dev/null 2>&1
        apt-get install -y libopencv-dev python3-opencv >/dev/null 2>&1
        python3 -c 'import cv2; print(f\"OpenCV version: {cv2.__version__}\")' 2>/dev/null && echo '✅ OpenCV安装成功'
    " || echo "❌ OpenCV测试失败"
    
    # 测试创建简单TensorRT网络
    echo "🧠 测试TensorRT网络创建..."
    docker run --rm --gpus all nvcr.io/nvidia/tensorrt:${version} python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
print('✅ TensorRT网络创建成功')
" 2>/dev/null || echo "❌ TensorRT网络创建失败"
}

# 主函数
main() {
    local command=${1:-"test"}
    local specific_version=$2
    
    echo "主机环境信息:"
    echo "CUDA驱动版本: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}' || echo '未检测到')"
    echo "Docker版本: $(docker --version || echo '未安装')"
    echo ""
    
    case $command in
        "test")
            if [ -n "$specific_version" ]; then
                test_version "$specific_version"
            else
                echo "测试推荐版本..."
                for version in "${VERSIONS[@]}"; do
                    if test_version "$version"; then
                        echo "✅ 版本 $version 测试通过"
                    else
                        echo "❌ 版本 $version 测试失败"
                    fi
                done
            fi
            ;;
        "detailed")
            if [ -z "$specific_version" ]; then
                echo "请指定要详细测试的版本，例如: $0 detailed 24.07-py3"
                exit 1
            fi
            test_version "$specific_version" && detailed_test "$specific_version"
            ;;
        "recommend")
            echo "🎯 根据你的CUDA 12.7环境，推荐版本:"
            echo ""
            echo "1. 首选: 24.07-py3 (TensorRT 10.2.0, CUDA 12.5) - 稳定推荐"
            echo "2. 最新: 24.08-py3 (TensorRT 10.3.0, CUDA 12.6) - 最新功能"
            echo "3. 保守: 24.05-py3 (TensorRT 10.0.1, CUDA 12.4) - 最稳定"
            echo ""
            echo "测试推荐版本:"
            test_version "24.07-py3"
            ;;
        *)
            echo "用法: $0 [command] [version]"
            echo ""
            echo "命令:"
            echo "  test [version]    - 测试版本兼容性 (默认测试所有推荐版本)"
            echo "  detailed <version> - 详细测试指定版本"
            echo "  recommend         - 显示推荐版本并测试"
            echo ""
            echo "示例:"
            echo "  $0 test                    # 测试所有推荐版本"
            echo "  $0 test 24.07-py3         # 测试特定版本"
            echo "  $0 detailed 24.07-py3     # 详细测试"
            echo "  $0 recommend               # 显示推荐版本"
            ;;
    esac
}

# 运行主函数
main "$@"
