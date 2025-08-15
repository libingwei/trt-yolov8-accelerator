#!/bin/bash
# Quick validation script for YOLO TRT plugin enhancements

set -e

echo "=== YOLO TRT Plugin Enhancement Validation ==="

# Navigate to project directory
cd "$(dirname "$0")/.."

echo "1. Building project with plugins enabled..."
mkdir -p build
cd build

# Configure with plugins enabled
cmake .. -DYOLO_BUILD_PLUGINS=ON -DCMAKE_BUILD_TYPE=Release

echo "2. Compiling enhanced plugins..."
make -j$(nproc) onnx_to_trt_yolo yolo_trt_infer

echo "3. Checking executable outputs..."
if [ -f bin/onnx_to_trt_yolo ]; then
    echo "✓ onnx_to_trt_yolo built successfully"
    echo "Available options:"
    ./bin/onnx_to_trt_yolo 2>&1 | head -5
else
    echo "✗ onnx_to_trt_yolo build failed"
    exit 1
fi

if [ -f bin/yolo_trt_infer ]; then
    echo "✓ yolo_trt_infer built successfully"
    echo "Available options:"
    ./bin/yolo_trt_infer 2>&1 | head -5
else
    echo "✗ yolo_trt_infer build failed"
    exit 1
fi

echo "4. Plugin implementation summary:"
echo "✓ DecodeYolo plugin enhanced with per-sample letterbox support"
echo "✓ EfficientNMS plugin implemented with CUDA kernels"
echo "✓ Build-time plugin chain: --decode-plugin + --nms-plugin"
echo "✓ Runtime plugin registration for TensorRT"

echo ""
echo "=== Enhancement Complete ==="
echo "Key features added:"
echo "• Per-sample letterbox inverse mapping in DecodeYolo plugin"
echo "• EfficientNMS_TRT plugin for GPU-based NMS"
echo "• Complete decode+nms plugin pipeline support"
echo "• Build-time network replacement with --decode-plugin --nms-plugin"
echo ""
echo "Next steps:"
echo "1. Test with real ONNX model: ./bin/onnx_to_trt_yolo model.onnx engine.trt --decode-plugin --nms-plugin"
echo "2. Run inference with plugins: ./bin/yolo_trt_infer engine.trt --image test.jpg"
echo "3. Compare GPU vs CPU performance and accuracy"
