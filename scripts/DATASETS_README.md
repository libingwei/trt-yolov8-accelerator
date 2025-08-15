# YOLOv8 Dataset Download and Usage Guide

## Quick Start

### 1. Download Datasets

```bash
# 从仓库根目录运行：
# Download all datasets (COCO + test images + ImageNet samples)
python3 projects/trt-yolov8-accelerator/scripts/download_yolov8_datasets.py --dataset all

# Download only COCO validation set
python3 projects/trt-yolov8-accelerator/scripts/download_yolov8_datasets.py --dataset coco

# Download only test images
python3 projects/trt-yolov8-accelerator/scripts/download_yolov8_datasets.py --dataset test

# Custom data directory and calibration size
python3 projects/trt-yolov8-accelerator/scripts/download_yolov8_datasets.py --data-dir /path/to/datasets --calib-images 200
```

### 2. Use with TensorRT Tools

```bash
# Build engine with INT8 calibration
./build/bin/onnx_to_trt_yolo model.onnx engine.trt --int8 --calib-dir ./datasets/calibration/coco_subset

# Build with plugins enabled
./build/bin/onnx_to_trt_yolo model.onnx engine.trt --fp16 --decode-plugin --nms-plugin

# Run inference on test image
./build/bin/yolo_trt_infer engine.trt --image ./datasets/yolov8_test/bus.jpg

# Run with plugin decoding
./build/bin/yolo_trt_infer engine.trt --image ./datasets/yolov8_test/bus.jpg --decode plugin
```

## Dataset Structure

After running the download script, you'll have:

```
# 默认目录结构（以 `--data-dir ./projects/trt-yolov8-accelerator/datasets` 为例）
projects/trt-yolov8-accelerator/datasets/
├── coco/
│   ├── val2017/           # 5,000 validation images
│   └── annotations/       # COCO annotations
├── yolov8_test/
│   ├── bus.jpg            # Test images
│   └── zidane.jpg
├── imagenet_sample/       # Sample ImageNet images
└── calibration/
  ├── coco_subset/       # 100 COCO images for INT8 calibration
  └── imagenet_subset/   # 100 ImageNet images for calibration
```

## Available Datasets

### 1. COCO 2017 Validation Set
- **Size**: ~1GB (5,000 images)
- **Use case**: Validation, accuracy testing
- **Classes**: 80 object categories
- **Source**: Official COCO dataset

### 2. YOLOv8 Test Images  
- **Size**: ~2MB (2 images)
- **Use case**: Quick functionality testing
- **Source**: Ultralytics official test images

### 3. ImageNet Sample
- **Size**: Variable (default 1,000 images)
- **Use case**: General purpose calibration
- **Source**: Public domain images (simplified)

### 4. Calibration Subsets
- **Size**: 100 images each (configurable)
- **Use case**: INT8 quantization calibration
- **Source**: Subsets from COCO and ImageNet

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | `./datasets` | Directory to store all datasets. When running from repository root we recommend `projects/trt-yolov8-accelerator/datasets` to keep files inside the project. |
| `--dataset` | `all` | Which dataset to download (`coco`, `imagenet`, `test`, `all`) |
| `--calib-images` | `100` | Number of images for calibration subset |
| `--imagenet-samples` | `1000` | Number of ImageNet sample images |

Note: COCO full validation needs ~1GB disk space. Ensure sufficient space and a stable network when downloading.
## Advanced Usage

### Custom Calibration Dataset

```bash
# Create calibration subset from your own images
python scripts/download_yolov8_datasets.py --dataset coco
python -c "
from download_yolov8_datasets import create_calibration_subset
create_calibration_subset('./my_images', './datasets/calibration/custom', 50)
"
```

### Batch Processing

```bash
# Process multiple images for validation
for img in ./datasets/coco/val2017/*.jpg; do
  ./build/bin/yolo_trt_infer engine.trt --image "$img"
done
```

## Troubleshooting

### Download Issues
- Check internet connection
- Verify disk space (COCO needs ~1GB)
- Some URLs might be temporary unavailable

### Calibration Issues  
- Ensure calibration images are representative of your use case
- More calibration images generally improve INT8 accuracy
- Use `--calib-images 200` for better results

### Performance Tips
- Use FP16 for best speed/accuracy balance
- INT8 provides maximum speed but requires good calibration data
- Plugin pipeline (`--decode-plugin --nms-plugin`) offers best GPU utilization

## Integration with Existing Workflows

This script is designed to work with:
- TensorRT engine building (`onnx_to_trt_yolo`)
- Inference testing (`yolo_trt_infer`) 
- INT8 calibration workflows
- YOLOv8 training and validation pipelines
