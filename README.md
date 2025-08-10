# trt-yolov8-accelerator (skeleton)

[变更日志 / Changelog](./CHANGELOG.md)

> 独立仓库最小骨架。你可以在此目录内 `git init` 并推送到 GitHub。

## 亮点（目标）
- YOLOv8 → ONNX → TensorRT（FP32/FP16/INT8），动态 batch/分辨率
- 自定义插件（NMS/Decode，`IPluginV2DynamicExt`）
- Nsight Systems/Compute 剖析与优化
- Triton 部署 + 客户端压测
- Docker 一键环境

## 目录结构
- src/, include/: 引擎加载、预处理、推理、后处理
- plugins/: TensorRT 插件与单元测试
- scripts/: ONNX 导出、评测、可视化
- benchmarks/: 硬件/配置与结果
- docker/: Dockerfile 与 compose
- triton/models/: 模型仓库与 config.pbtxt
- docs/: 报告、图表、设计说明

## 起步
```bash
# 在该目录内初始化为独立 git 仓库
# git init
# git add .
# git commit -m "init skeleton"
# git remote add origin <your-repo-url>
# git push -u origin main
```

## 使用（最小流程）
```bash
# 1) 安装 Python 依赖（导出/验证 ONNX）
pip install -r requirements.txt

# 2) 导出 YOLOv8 ONNX（默认 yolov8n.pt，需要可自备权重）
python scripts/export_yolov8_onnx.py --weights yolov8n.pt --outdir models --imgsz 640

# 3) 构建 Docker（包含 TensorRT 与依赖）
docker build -t trt-yolov8-accelerator:dev -f docker/Dockerfile .

# 4) 进入容器后构建与运行占位程序
# docker run --gpus all -it --rm -v $PWD:/workspace trt-yolov8-accelerator:dev bash
# ./build/bin/onnx_to_trt_yolo
# ./build/bin/yolo_trt_infer
```

说明：导出脚本默认优先使用 GPU（cuda:0），若不可用或失败会自动回退到 CPU。可手动指定：
```bash
python scripts/export_yolov8_onnx.py --weights yolov8n.pt --device cuda:0
python scripts/export_yolov8_onnx.py --weights yolov8n.pt --device cpu
```

## 推理与可视化
```bash
# 将导出的 ONNX 转为 TensorRT 引擎（FP16 示例）
./build/bin/onnx_to_trt_yolo models/yolov8n.onnx models/yolov8n_fp16.trt --fp16 --min 1x3x320x320 --opt 1x3x640x640 --max 16x3x1280x1280

# 运行推理并在当前目录生成 yolo_out.jpg
./build/bin/yolo_trt_infer models/yolov8n_fp16.trt --image assets/sample.jpg --H 640 --W 640 --conf 0.25 --iou 0.5
```

注意：
- 输出维度目前按 YOLOv8 导出默认布局解析（[N, max_det, 4+obj+num_classes] 或 [max_det, 4+... ]）。若你的导出图不同，请在 `src/yolo_trt_infer.cpp` 中调整解析逻辑或先使用 CPU NMS 验证。
- 若未提供 `--image`，程序仅跑一次前向并打印张量尺寸。
