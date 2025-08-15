# YOLOv8 视频追踪和计数系统

基于TensorRT加速的YOLOv8实时视频目标检测、追踪和计数系统。

## 🚀 功能特性

- **实时目标检测**: 基于YOLOv8和TensorRT的高性能目标检测
- **多目标追踪**: 基于IoU匹配的简单而有效的追踪算法
- **区域计数**: 支持多边形区域的进入/离开计数
- **线段计数**: 支持跨越指定线段的计数
- **实时可视化**: 显示检测框、追踪轨迹、ID和统计信息
- **配置化设计**: 支持配置文件灵活设置参数
- **性能优化**: 支持多种精度模式和参数调优

## 📁 目录结构

```
video_tracking/
├── include/                    # 头文件
│   ├── object_tracker.h       # 目标追踪器
│   ├── zone_counter.h          # 区域计数器
│   ├── video_processor.h       # 视频处理器
│   └── tracking_config.h       # 配置管理
├── src/                        # 源文件
│   ├── object_tracker.cpp
│   ├── zone_counter.cpp
│   ├── video_processor.cpp
│   └── tracking_config.cpp
├── test/                       # 测试文件
│   ├── basic_test.cpp          # 基础功能测试
│   └── advanced_test.cpp       # 高级可视化测试
├── test_videos/               # 测试视频目录
├── config.txt                 # 配置文件
├── run_demo.sh               # 演示脚本
├── verify_setup.sh           # 验证脚本
└── README.md                 # 说明文档
```

## 🛠 依赖要求

### 必需依赖
- **OpenCV 4.x**: 图像处理和视频I/O
- **C++17编译器**: GCC 7+ 或 Clang 5+
- **CMake 3.18+**: 构建系统

### TensorRT功能依赖（可选）
- **CUDA Toolkit**: GPU加速
- **TensorRT**: 推理引擎
- **主项目的trt_utils库**: TensorRT封装

## 🔧 构建方法

### 方法1: 作为主项目的一部分构建（推荐）

```bash
# 在主项目根目录
cd /Users/koo/Code/diy/AI-Infer-Acc
mkdir build && cd build
cmake .. -DYOLO_BUILD_PLUGINS=OFF
make -j$(nproc)
```

### 方法2: 独立构建（仅基础功能）

```bash
cd video_tracking
mkdir build_standalone && cd build_standalone
cmake .. -f ../CMakeLists_standalone.txt
make -j$(nproc)
```

### 方法3: 使用验证脚本一键构建

```bash
cd video_tracking
chmod +x verify_setup.sh
./verify_setup.sh
```

## 🚀 快速开始

### 1. 运行基础测试

```bash
# 运行基础追踪功能测试
./build_standalone/bin/tracking_test

# 运行高级可视化测试
./build_standalone/bin/advanced_test
```

### 2. 使用演示脚本

```bash
chmod +x run_demo.sh
./run_demo.sh
```

### 3. 准备YOLOv8模型（需要TensorRT支持）

```bash
# 在主项目目录
cd /Users/koo/Code/diy/AI-Infer-Acc/projects/trt-yolov8-accelerator

# 安装Python依赖
pip install -r requirements.txt

# 导出ONNX模型
python scripts/export_yolov8_onnx.py --weights yolov8n.pt --outdir models --imgsz 640

# 转换为TensorRT引擎
./build/bin/onnx_to_trt_yolo models/yolov8n.onnx models/yolov8n.trt --fp16
```

### 4. 处理视频（需要TensorRT支持）

```bash
# 处理视频文件
./build/bin/video_tracker models/yolov8n.trt --video test_videos/sample.mp4 --output output.mp4

# 处理摄像头流
./build/bin/video_tracker models/yolov8n.trt --camera 0 --output live_output.mp4

# 自定义参数
./build/bin/video_tracker models/yolov8n.trt \\
    --video input.mp4 \\
    --output output.mp4 \\
    --width 640 \\
    --height 640 \\
    --conf 0.3 \\
    --iou 0.5
```

## ⚙️ 配置系统

### 配置文件示例 (config.txt)

```ini
# 模型配置
engine_path=models/yolov8n.trt
input_width=640
input_height=640
conf_threshold=0.25
iou_threshold=0.5

# 追踪器配置
tracker_iou_threshold=0.5
max_missed_frames=10

# 显示配置
show_detection_boxes=true
show_tracking_boxes=true
show_trajectories=true

# 输出配置
output_path=output.mp4
```

### 配置计数区域

在`video_processor.cpp`的`setupCountingZones()`方法中配置：

```cpp
void VideoProcessor::setupCountingZones() {
    // 添加矩形计数区域
    std::vector<cv::Point> zone1 = {
        cv::Point(100, 100),  // 左上角
        cv::Point(500, 100),  // 右上角
        cv::Point(500, 400),  // 右下角
        cv::Point(100, 400)   // 左下角
    };
    counter_.addCountingZone(CountingZone(zone1, "Zone1", cv::Scalar(0, 255, 0)));
    
    // 添加计数线
    counter_.addCrossingLine(CrossingLine(
        cv::Point(300, 50),   // 起点
        cv::Point(300, 600),  // 终点
        "Line1", 
        cv::Scalar(255, 0, 0)
    ));
}
```

## 🎮 交互控制

运行时按键控制：
- `q` 或 `ESC`: 退出程序
- `r`: 重置计数器和追踪器

## 📊 性能优化建议

1. **输入分辨率**: 根据需要调整输入分辨率，较小分辨率可提高FPS
2. **置信度阈值**: 提高置信度阈值可减少误检，提高追踪稳定性
3. **追踪参数**: 调整IoU阈值和最大丢失帧数以适应不同场景
4. **TensorRT优化**: 使用FP16或INT8精度以提高推理速度
5. **轨迹长度**: 限制轨迹历史长度以节省内存

## 🧪 测试功能

### 基础测试
- 追踪器IoU匹配测试
- 区域计数逻辑测试
- 线段穿越检测测试

### 高级测试
- 实时可视化追踪
- 多对象模拟测试
- 性能基准测试
- 交互式参数调试

### 性能基准
- 1000帧处理速度测试
- 内存使用情况监控
- 不同参数配置的性能对比

## 🔧 扩展开发

### 添加新的追踪算法
在`ObjectTracker`类中实现新的匹配算法：
```cpp
// 示例：基于卡尔曼滤波的预测
class KalmanTracker : public ObjectTracker {
    // 实现预测和更新逻辑
};
```

### 添加更复杂的计数逻辑
在`ZoneCounter`类中添加新功能：
```cpp
// 示例：方向检测、停留时间统计
void addDirectionDetection(const std::string& zone_name);
void addDwellTimeAnalysis(float min_dwell_time);
```

### 集成其他检测模型
修改`VideoProcessor`类以支持其他模型：
```cpp
// 支持YOLOv5, YOLOv7等其他模型
class MultiModelProcessor : public VideoProcessor {
    // 实现多模型支持
};
```

## 🐛 故障排除

### 编译错误
1. **OpenCV未找到**: 
   ```bash
   # macOS
   brew install opencv
   # Ubuntu
   sudo apt-get install libopencv-dev
   ```

2. **CMake版本过低**: 升级到3.18+
3. **C++17支持**: 确保编译器支持C++17标准

### 运行时错误
1. **模型文件路径**: 检查engine_path是否正确
2. **视频格式**: 确保输入视频格式被OpenCV支持
3. **GPU内存**: 验证GPU内存是否充足

### 性能问题
1. **监控GPU使用率**: 使用nvidia-smi监控
2. **检查输入分辨率**: 过大的分辨率会影响性能
3. **优化模型精度**: 使用FP16或INT8模式

## 📝 API文档

### ObjectTracker 类

```cpp
class ObjectTracker {
public:
    ObjectTracker(float iou_threshold = 0.5f, int max_missed_frames = 10);
    
    // 更新追踪器，输入新的检测结果
    void update(const std::vector<Detection>& detections);
    
    // 获取当前活跃的追踪对象
    std::vector<TrackedObject> getActiveObjects() const;
    
    // 重置追踪器
    void reset();
};
```

### ZoneCounter 类

```cpp
class ZoneCounter {
public:
    // 添加计数区域
    void addCountingZone(const CountingZone& zone);
    
    // 添加计数线
    void addCrossingLine(const CrossingLine& line);
    
    // 更新计数
    void updateCounts(const std::vector<TrackedObject>& tracked_objects);
    
    // 获取统计结果
    std::map<std::string, std::pair<int, int>> getZoneCounts() const;
    std::map<std::string, int> getLineCounts() const;
};
```

### VideoProcessor 类

```cpp
class VideoProcessor {
public:
    VideoProcessor(const std::string& engine_path, 
                   int input_width = 640, int input_height = 640,
                   float conf_threshold = 0.25f, float iou_threshold = 0.5f);
    
    // 初始化处理器
    bool initialize();
    
    // 处理视频文件
    bool processVideo(const std::string& video_path, 
                     const std::string& output_path = "",
                     bool show_preview = true);
    
    // 处理摄像头输入
    bool processCameraStream(int camera_id = 0, 
                           const std::string& output_path = "");
};
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目遵循主项目许可证。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - 推理加速引擎
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发起 Discussion
- 查看项目 Wiki

---

**注意**: 这是一个持续开发的项目，部分功能可能需要进一步完善和优化。
