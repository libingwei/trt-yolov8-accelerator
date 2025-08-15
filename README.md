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

下面的示例假定你在仓库根目录（也就是 `/workspace/AI-Infer-Acc`）执行命令；若你切换到项目子目录，请相应调整路径。

1) 安装 Python 依赖（用于导出/验证 ONNX）

```bash
# 从仓库根目录运行
pip install -r projects/trt-yolov8-accelerator/requirements.txt
```

2) 导出 YOLOv8 为 ONNX

```bash
# 使用 ultralytics 导出（默认从 weights 同级生成 .onnx）
# 在仓库根目录运行：
python3 projects/trt-yolov8-accelerator/scripts/export_yolov8_onnx.py --weights yolov8n.pt --outdir projects/trt-yolov8-accelerator/models --imgsz 640

# 如果在项目子目录运行（projects/trt-yolov8-accelerator）：
python3 scripts/export_yolov8_onnx.py --weights yolov8n.pt --outdir models --imgsz 640
```

备注：导出脚本会优先使用 GPU (cuda:0) 如果可用；若导出失败会回退到 CPU。你也可以显式指定设备：

```bash
# 指定 GPU
python3 projects/trt-yolov8-accelerator/scripts/export_yolov8_onnx.py --weights yolov8n.pt --device cuda:0
# 指定 CPU
python3 projects/trt-yolov8-accelerator/scripts/export_yolov8_onnx.py --weights yolov8n.pt --device cpu
```

3) 下载测试数据与标定数据（可选）

```bash
# 从仓库根目录运行，默认下载到 projects/trt-yolov8-accelerator/datasets
python3 projects/trt-yolov8-accelerator/scripts/download_yolov8_datasets.py --data-dir projects/trt-yolov8-accelerator/datasets --dataset all
```

注意：COCO 全量数据较大（约 1GB），如果只做快速验证可只下载 `--dataset test` 或 `--dataset imagenet` 的样本。

4) 构建并运行（TensorRT 环境通常在容器或带有 TensorRT 的系统上）

```bash
# 构建 Docker（可选）
docker build -t trt-yolov8-accelerator:dev -f projects/trt-yolov8-accelerator/docker/Dockerfile .

# 构建工程（在容器或系统上，示例：在仓库根目录）
cmake -S . -B build
cmake --build build -- -j4

# 示例：使用已构建的转换与推理二进制（路径相对于仓库根目录）
./build/bin/onnx_to_trt_yolo projects/trt-yolov8-accelerator/models/yolov8n.onnx projects/trt-yolov8-accelerator/models/yolov8n_fp16.trt --fp16 --min 1x3x320x320 --opt 1x3x640x640 --max 16x3x1280x1280

./build/bin/yolo_trt_infer projects/trt-yolov8-accelerator/models/yolov8n_fp16.trt --image projects/trt-yolov8-accelerator/assets/sample.jpg --H 640 --W 640 --conf 0.25 --iou 0.5
```

输出：
```bash
Results saved to /content/drive/MyDrive/AI-Infer-Acc
Predict:         yolo predict task=detect model=yolov8n.onnx imgsz=640  
Validate:        yolo val task=detect model=yolov8n.onnx imgsz=640 data=coco.yaml  
Visualize:       https://netron.app
Exported: models/yolov8n.onnx
Input: images ['batch', 3, 'height', 'width'] tensor(float)
Outputs:
  - output0 ['batch', 'Concatoutput0_dim_1', 'anchors'] tensor(float)
```

## INT8 标定（可选）

环境变量开关：
- CALIB_RECURSIVE=1：递归收集标定目录下所有子目录里的图片（默认关闭，仅扫描根目录）
- IMAGENET_CENTER_CROP=1：开启“短边 256 + 中心裁剪到 HxW”的预处理（默认关闭）
- YOLO_MEAN/YOLO_STD：逗号分隔三通道值，设置后将启用 (img-mean)/std 归一化；未设置时默认仅缩放到 [0,1]

示例：
```bash
# 使用自备标定集目录进行 INT8 标定导出（默认不做归一化，仅 [0,1] + RGB）
CALIB_RECURSIVE=1 \
./build/bin/onnx_to_trt_yolo models/yolov8n.onnx models/yolov8n_int8.trt --int8 --calib-dir calibration_data

# 如需与特定推理链路对齐的归一化（示例为 identity）
YOLO_MEAN=0,0,0 YOLO_STD=1,1,1 CALIB_RECURSIVE=1 \
./build/bin/onnx_to_trt_yolo models/yolov8n.onnx models/yolov8n_int8.trt --int8 --calib-dir calibration_data
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

### 运行时开关（解码与 NMS）

`yolo_trt_infer` 支持以下与后处理相关的参数：

- `--decode cpu|plugin`：选择解码路径（默认 `cpu`；`plugin` 表示解码在图内由插件完成，本程序不再做 CPU 解码）。
- `--has-nms`：当导出的 ONNX 已包含 NMS（例如 `--nms` 导出）时，跳过本地解码与 NMS，直接绘制 [x1,y1,x2,y2,conf,cls]。
- `--class-agnostic`：使用 class-agnostic NMS（默认开启）。
- `--topk N`：NMS 后保留的最大框数（默认无限制）。

示例：
```bash
# 已融合 NMS 的 ONNX（导出时使用 --nms）
./build/bin/yolo_trt_infer models/yolov8n_fp16.trt --image assets/sample.jpg --H 640 --W 640 --has-nms

# 使用 CPU 解码 + NMS，并限制 TopK
./build/bin/yolo_trt_infer models/yolov8n_fp16.trt --image assets/sample.jpg --H 640 --W 640 --decode cpu --topk 100
```

### 构建期开关（插件占位）

- CMake 选项：`-DYOLO_BUILD_PLUGINS=ON` 将构建解码插件目标（目前为占位骨架，便于后续实现 CUDA kernel）。
- 转换工具：`onnx_to_trt_yolo` 提供 `--decode-plugin`，在构建期将 YOLO 头部替换为解码插件层（需配合 `-DYOLO_BUILD_PLUGINS=ON`）。

说明：插件路径用于与 CPU 解码做 A/B 对比，确认数值一致性与性能差异。实际 CUDA 解码 kernel 与网络替换逻辑将分阶段补全。
