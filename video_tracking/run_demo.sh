#!/bin/bash

# YOLOv8 视频追踪演示脚本

echo "=== YOLOv8 视频追踪系统演示 ==="

TRACKING_DIR="/Users/koo/Code/diy/AI-Infer-Acc/projects/trt-yolov8-accelerator/video_tracking"
cd "$TRACKING_DIR"

# 检查是否已构建
if [ ! -f "build_standalone/bin/tracking_test" ]; then
    echo "未找到构建结果，开始构建..."
    mkdir -p build_standalone
    cd build_standalone
    cmake .. -f ../CMakeLists_standalone.txt
    make -j$(nproc)
    cd ..
fi

echo "可用的演示选项："
echo "1. 基础追踪测试 (不需要GPU)"
echo "2. 高级追踪测试 (包含可视化)"
echo "3. 性能基准测试"
echo "4. 配置文件展示"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "运行基础追踪测试..."
        ./build_standalone/bin/tracking_test
        ;;
    2)
        echo "运行高级追踪测试..."
        if [ -f "build_standalone/bin/advanced_test" ]; then
            ./build_standalone/bin/advanced_test
        else
            echo "高级测试程序未构建，请检查CMakeLists.txt"
        fi
        ;;
    3)
        echo "运行性能测试..."
        echo "测试1000帧的处理性能..."
        time ./build_standalone/bin/tracking_test
        ;;
    4)
        echo "展示配置文件内容:"
        cat config.txt
        echo ""
        echo "配置文件位置: $TRACKING_DIR/config.txt"
        echo "您可以根据需要修改此文件"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "演示完成！"
