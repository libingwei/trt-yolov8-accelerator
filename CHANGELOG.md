# Changelog

本文件按时间顺序记录子项目的每次改动，便于复现与追踪。格式参考 Keep a Changelog。

## [2025-08-10]
### Added
- 推理端到端完善：
  - `src/yolo_trt_infer.cpp` 增加图像预处理（letterbox、BGR→RGB、归一化、CHW）。
  - 兼容多种 YOLOv8 输出布局（[B,N,C] / [B,C,N] / [N,C] / [C,N]），支持未 NMS 输出的解码与 CPU NMS，及已 NMS 输出的直绘。
  - 保存可视化结果到 `yolo_out.jpg`；新增命令行参数：`--image`、`--H`、`--W`、`--conf`、`--iou`。
- 文档：
  - `README.md` 增加 ONNX→TRT 转换与推理可视化示例命令，说明输出布局兼容策略与注意事项。

### Known
- 解析逻辑基于通用布局推断，个别导出图可能需微调列含义；后续将以 TensorRT 插件实现 GPU 端 Decode+NMS。

## [2025-08-09]
### Added
- 初始骨架：
  - `CMakeLists.txt`：配置 CUDA/TensorRT/OpenCV，生成 `onnx_to_trt_yolo` 与 `yolo_trt_infer` 两个目标。
  - `docker/Dockerfile`：基于 NVIDIA TensorRT 镜像，安装构建依赖与 Python 依赖。
  - `scripts/export_yolov8_onnx.py`：YOLOv8 导出 ONNX，GPU 优先，失败回退 CPU，含 ORT 校验。
  - `src/yolo_int8_calibrator.cpp`：INT8 标定实现（Entropy）与缓存机制。
  - `src/onnx_to_trt_yolo.cpp`：ONNX 解析、动态 profile、FP16/INT8 构建与引擎序列化。
  - `src/yolo_trt_infer.cpp`：最小化推理骨架（加载引擎、设置动态输入、单次前向）。
