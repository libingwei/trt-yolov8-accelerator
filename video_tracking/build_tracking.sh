#!/bin/bash

# YOLOv8 视频追踪系统构建脚本
# 使用方法: ./build_tracking.sh [clean|test|full]

set -e

PROJECT_ROOT="/Users/koo/Code/diy/AI-Infer-Acc"
YOLO_PROJECT_DIR="$PROJECT_ROOT/projects/trt-yolov8-accelerator"
BUILD_DIR="$PROJECT_ROOT/build"

echo "=== YOLOv8 视频追踪系统构建脚本 ==="
echo "项目根目录: $PROJECT_ROOT"
echo "构建目录: $BUILD_DIR"

# 检查参数
ACTION=${1:-"build"}

case $ACTION in
    "clean")
        echo "清理构建目录..."
        rm -rf "$BUILD_DIR"
        echo "清理完成"
        exit 0
        ;;
    "test")
        echo "运行基础测试..."
        if [ ! -f "$BUILD_DIR/projects/trt-yolov8-accelerator/video_tracking/bin/tracking_test" ]; then
            echo "错误: 测试程序不存在，请先构建项目"
            exit 1
        fi
        cd "$BUILD_DIR/projects/trt-yolov8-accelerator/video_tracking/bin"
        ./tracking_test
        exit 0
        ;;
    "full")
        echo "完整构建（包括插件）..."
        PLUGINS_FLAG="-DYOLO_BUILD_PLUGINS=ON"
        ;;
    *)
        echo "标准构建..."
        PLUGINS_FLAG="-DYOLO_BUILD_PLUGINS=OFF"
        ;;
esac

# 检查依赖
echo "检查依赖..."

# 检查OpenCV
if ! pkg-config --exists opencv4; then
    echo "警告: OpenCV4未找到，尝试查找opencv..."
    if ! pkg-config --exists opencv; then
        echo "错误: 未找到OpenCV，请先安装OpenCV"
        exit 1
    fi
fi

echo "✓ OpenCV已找到"

# 检查CUDA（可选）
if command -v nvcc >/dev/null 2>&1; then
    echo "✓ CUDA已找到: $(nvcc --version | grep release | cut -d' ' -f6)"
else
    echo "⚠ CUDA未找到，将跳过TensorRT功能"
fi

# 创建构建目录
echo "创建构建目录..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置CMake
echo "配置CMake..."
if command -v nvcc >/dev/null 2>&1; then
    # 有CUDA支持
    cmake "$PROJECT_ROOT" $PLUGINS_FLAG
else
    # 无CUDA支持，只构建基础功能
    echo "无CUDA环境，只构建基础追踪功能..."
    cd "$YOLO_PROJECT_DIR/video_tracking"
    mkdir -p build && cd build
    cmake .. -f ../CMakeLists_standalone.txt
    make -j$(nproc)
    echo "基础追踪模块构建完成"
    echo "运行测试: cd $YOLO_PROJECT_DIR/video_tracking/build && ./bin/tracking_test"
    exit 0
fi

# 构建项目
echo "开始构建..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 构建完成！"
    echo ""
    echo "可执行程序位置:"
    echo "  - 基础YOLOv8推理: $BUILD_DIR/projects/trt-yolov8-accelerator/bin/yolo_trt_infer"
    echo "  - ONNX转TensorRT: $BUILD_DIR/projects/trt-yolov8-accelerator/bin/onnx_to_trt_yolo"
    echo "  - 追踪测试: $BUILD_DIR/projects/trt-yolov8-accelerator/video_tracking/bin/tracking_test"
    echo ""
    echo "下一步操作:"
    echo "1. 运行基础测试: ./build_tracking.sh test"
    echo "2. 准备YOLOv8模型:"
    echo "   cd $YOLO_PROJECT_DIR"
    echo "   pip install -r requirements.txt"
    echo "   python scripts/export_yolov8_onnx.py --weights yolov8n.pt --outdir models"
    echo "   $BUILD_DIR/projects/trt-yolov8-accelerator/bin/onnx_to_trt_yolo models/yolov8n.onnx models/yolov8n.trt --fp16"
    echo ""
else
    echo "❌ 构建失败"
    exit 1
fi
