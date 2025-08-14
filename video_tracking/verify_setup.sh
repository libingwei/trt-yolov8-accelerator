#!/bin/bash

# YOLOv8 视频追踪系统验证脚本
set -e

echo "=== YOLOv8 视频追踪系统验证 ==="

# 检查基础依赖
echo "检查基础依赖..."

# 检查OpenCV
if pkg-config --exists opencv4; then
    echo "✓ OpenCV4 已安装: $(pkg-config --modversion opencv4)"
elif pkg-config --exists opencv; then
    echo "✓ OpenCV 已安装: $(pkg-config --modversion opencv)"
else
    echo "❌ OpenCV 未找到，请先安装 OpenCV"
    echo "macOS: brew install opencv"
    echo "Ubuntu: sudo apt-get install libopencv-dev"
    exit 1
fi

# 检查CMake
if command -v cmake >/dev/null 2>&1; then
    echo "✓ CMake 已安装: $(cmake --version | head -n1 | cut -d' ' -f3)"
else
    echo "❌ CMake 未找到，请先安装 CMake"
    exit 1
fi

# 检查编译器
if command -v g++ >/dev/null 2>&1; then
    echo "✓ g++ 已安装: $(g++ --version | head -n1)"
elif command -v clang++ >/dev/null 2>&1; then
    echo "✓ clang++ 已安装: $(clang++ --version | head -n1)"
else
    echo "❌ 未找到 C++ 编译器"
    exit 1
fi

PROJECT_ROOT="/Users/koo/Code/diy/AI-Infer-Acc"
YOLO_DIR="$PROJECT_ROOT/projects/trt-yolov8-accelerator"
TRACKING_DIR="$YOLO_DIR/video_tracking"

echo "项目目录: $TRACKING_DIR"

# 进入追踪模块目录
cd "$TRACKING_DIR"

# 第一步：独立构建基础追踪功能
echo "=== 第一步：构建基础追踪功能 ==="
mkdir -p build_standalone
cd build_standalone

echo "配置CMake..."
cmake .. -f ../CMakeLists_standalone.txt

echo "编译基础追踪库..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "✓ 基础追踪库编译成功"
else
    echo "❌ 基础追踪库编译失败"
    exit 1
fi

# 运行基础测试
echo "=== 第二步：运行基础测试 ==="
if [ -f "./bin/tracking_test" ]; then
    echo "运行基础追踪测试..."
    ./bin/tracking_test
    if [ $? -eq 0 ]; then
        echo "✓ 基础测试通过"
    else
        echo "❌ 基础测试失败"
        exit 1
    fi
else
    echo "❌ 测试程序未找到"
    exit 1
fi

# 第三步：尝试在主项目中构建
echo "=== 第三步：在主项目中构建 ==="
cd "$PROJECT_ROOT"

# 检查是否存在构建目录
if [ -d "build" ]; then
    echo "清理旧的构建目录..."
    rm -rf build
fi

mkdir -p build
cd build

echo "配置主项目CMake..."
if cmake .. -DYOLO_BUILD_PLUGINS=OFF; then
    echo "✓ CMake 配置成功"
else
    echo "❌ CMake 配置失败"
    exit 1
fi

echo "编译主项目..."
if make -j$(nproc); then
    echo "✓ 主项目编译成功"
    
    # 检查生成的可执行文件
    echo "检查生成的可执行文件："
    if [ -f "projects/trt-yolov8-accelerator/video_tracking/bin/tracking_test" ]; then
        echo "✓ tracking_test 已生成"
    fi
    
    if [ -f "projects/trt-yolov8-accelerator/video_tracking/bin/video_tracker" ]; then
        echo "✓ video_tracker 已生成"
    else
        echo "⚠ video_tracker 未生成（可能缺少TensorRT依赖）"
    fi
    
    # 运行主项目中的测试
    echo "运行主项目中的追踪测试..."
    ./projects/trt-yolov8-accelerator/video_tracking/bin/tracking_test
    
else
    echo "❌ 主项目编译失败"
    exit 1
fi

echo ""
echo "🎉 验证完成！"
echo ""
echo "总结："
echo "✓ 基础追踪功能正常"
echo "✓ 编译环境正常"
echo "✓ 可以开始开发和测试"
echo ""
echo "下一步操作："
echo "1. 准备测试视频放入 $TRACKING_DIR/test_videos/"
echo "2. 如果有TensorRT环境，可以测试完整的视频处理功能"
echo "3. 根据需要修改计数区域配置"
echo ""
echo "构建输出位置："
echo "- 独立构建: $TRACKING_DIR/build_standalone/bin/"
echo "- 主项目构建: $PROJECT_ROOT/build/projects/trt-yolov8-accelerator/video_tracking/bin/"
