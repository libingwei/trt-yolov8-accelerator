# trt-yolov8-accelerator (skeleton)

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
