#include "../include/video_processor.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

class Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* m) noexcept override {
        if (s <= Severity::kWARNING) {
            std::cout << m << "\n";
        }
    }
};

VideoProcessor::VideoProcessor(const std::string& engine_path,
                             int input_width,
                             int input_height,
                             float conf_threshold,
                             float iou_threshold)
    : engine_path_(engine_path),
      input_width_(input_width),
      input_height_(input_height),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold),
      tracker_(0.5f, 10) {
    
    logger_ = std::make_unique<Logger>();
    initializeClassNames();
}

bool VideoProcessor::initialize() {
    runner_ = std::make_unique<TrtRunner>(*logger_);
    
    if (!runner_->loadEngineFromFile(engine_path_)) {
        std::cerr << "Failed to load engine from " << engine_path_ << std::endl;
        return false;
    }
    
    if (!runner_->prepare(1, input_height_, input_width_)) {
        std::cerr << "Failed to prepare runner" << std::endl;
        return false;
    }
    
    std::cout << "Video processor initialized successfully" << std::endl;
    return true;
}

std::vector<Detection> VideoProcessor::processFrame(const cv::Mat& frame) {
    LetterboxInfo letterbox_info;
    cv::Mat preprocessed = preprocessFrame(frame, letterbox_info);
    
    // 准备输入数据
    size_t input_size = runner_->inputSize();
    std::vector<float> input_data(input_size);
    
    // 转换为CHW格式
    std::vector<cv::Mat> channels;
    cv::split(preprocessed, channels);
    size_t plane_size = input_width_ * input_height_;
    
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data.data() + c * plane_size, 
               channels[c].data, 
               plane_size * sizeof(float));
    }
    
    // 运行推理
    size_t output_size = runner_->outputSize();
    std::vector<float> output_data(output_size);
    
    runner_->run(1, input_data.data(), output_data.data(), true);
    
    // 后处理
    auto output_dims = runner_->outputDims();
    return postprocessResults(output_data, output_dims, letterbox_info, 
                            frame.cols, frame.rows);
}

bool VideoProcessor::processVideo(const std::string& video_path,
                                const std::string& output_path,
                                bool show_preview) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video: " << video_path << std::endl;
        return false;
    }
    
    // 获取视频属性
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    cv::VideoWriter writer;
    if (!output_path.empty()) {
        writer.open(output_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                   fps, cv::Size(frame_width, frame_height));
        if (!writer.isOpened()) {
            std::cerr << "Error opening output video writer" << std::endl;
            return false;
        }
    }
    
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (cap.read(frame)) {
        frame_count++;
        
        // 处理帧
        auto detections = processFrame(frame);
        
        // 更新追踪器
        tracker_.update(detections);
        auto tracked_objects = tracker_.getActiveObjects();
        
        // 更新计数器
        counter_.updateCounts(tracked_objects);
        
        // 绘制结果
        drawResults(frame, detections, tracked_objects);
        counter_.drawZones(frame);
        drawStatistics(frame);
        
        // 保存输出
        if (writer.isOpened()) {
            writer.write(frame);
        }
        
        // 显示预览
        if (show_preview) {
            cv::imshow("Video Tracking", frame);
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) { // ESC键
                break;
            } else if (key == 'r') {
                // 重置计数
                counter_.resetCounts();
                tracker_.reset();
            }
        }
        
        // 显示进度
        if (frame_count % 30 == 0) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
            double current_fps = frame_count * 1000.0 / elapsed;
            std::cout << "Processed " << frame_count << " frames, FPS: " 
                     << std::fixed << std::setprecision(2) << current_fps << std::endl;
        }
    }
    
    // 输出最终统计
    std::cout << "\nFinal Statistics:" << std::endl;
    auto zone_counts = counter_.getZoneCounts();
    for (const auto& pair : zone_counts) {
        std::cout << "Zone " << pair.first << ": Enter=" << pair.second.first 
                 << ", Exit=" << pair.second.second << std::endl;
    }
    
    auto line_counts = counter_.getLineCounts();
    for (const auto& pair : line_counts) {
        std::cout << "Line " << pair.first << ": Crossings=" << pair.second << std::endl;
    }
    
    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    return true;
}

bool VideoProcessor::processCameraStream(int camera_id, const std::string& output_path) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera: " << camera_id << std::endl;
        return false;
    }
    
    // 设置摄像头属性
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);
    
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    cv::VideoWriter writer;
    if (!output_path.empty()) {
        writer.open(output_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                   fps, cv::Size(frame_width, frame_height));
    }
    
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    std::cout << "Press 'q' to quit, 'r' to reset counters" << std::endl;
    
    while (cap.read(frame)) {
        frame_count++;
        
        // 处理帧
        auto detections = processFrame(frame);
        
        // 更新追踪器
        tracker_.update(detections);
        auto tracked_objects = tracker_.getActiveObjects();
        
        // 更新计数器
        counter_.updateCounts(tracked_objects);
        
        // 绘制结果
        drawResults(frame, detections, tracked_objects);
        counter_.drawZones(frame);
        drawStatistics(frame);
        
        // 保存输出
        if (writer.isOpened()) {
            writer.write(frame);
        }
        
        // 显示实时画面
        cv::imshow("Live Tracking", frame);
        char key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            break;
        } else if (key == 'r') {
            counter_.resetCounts();
            tracker_.reset();
            std::cout << "Counters and tracker reset" << std::endl;
        }
        
        // 显示FPS
        if (frame_count % 30 == 0) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
            double current_fps = frame_count * 1000.0 / elapsed;
            std::cout << "FPS: " << std::fixed << std::setprecision(2) 
                     << current_fps << std::endl;
        }
    }
    
    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    return true;
}

void VideoProcessor::setupCountingZones() {
    // 示例：添加一个矩形计数区域
    std::vector<cv::Point> zone1 = {
        cv::Point(100, 100),
        cv::Point(500, 100),
        cv::Point(500, 400),
        cv::Point(100, 400)
    };
    counter_.addCountingZone(CountingZone(zone1, "Zone1", cv::Scalar(0, 255, 0)));
    
    // 示例：添加一条计数线
    counter_.addCrossingLine(CrossingLine(cv::Point(300, 50), cv::Point(300, 600), 
                                        "Line1", cv::Scalar(255, 0, 0)));
}

void VideoProcessor::drawResults(cv::Mat& frame,
                               const std::vector<Detection>& detections,
                               const std::vector<TrackedObject>& tracked_objects) {
    // 绘制检测框
    for (const auto& det : detections) {
        cv::rectangle(frame, det.bbox, cv::Scalar(255, 255, 0), 2);
    }
    
    // 绘制追踪对象
    for (const auto& obj : tracked_objects) {
        if (!obj.is_active) continue;
        
        // 绘制边界框
        cv::Scalar color(0, 255, 0);
        cv::rectangle(frame, obj.bbox, color, 2);
        
        // 绘制ID和类别
        std::string label = "ID:" + std::to_string(obj.id);
        if (obj.class_id < class_names_.size()) {
            label += " " + class_names_[obj.class_id];
        }
        label += " " + std::to_string(static_cast<int>(obj.confidence * 100)) + "%";
        
        cv::Point label_pos(obj.bbox.x, obj.bbox.y - 10);
        cv::putText(frame, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        
        // 绘制轨迹
        if (obj.trajectory.size() > 1) {
            for (size_t i = 1; i < obj.trajectory.size(); ++i) {
                cv::line(frame, obj.trajectory[i-1], obj.trajectory[i], 
                        cv::Scalar(0, 0, 255), 2);
            }
        }
        
        // 绘制中心点
        cv::circle(frame, obj.center, 3, cv::Scalar(0, 0, 255), -1);
    }
}

void VideoProcessor::drawStatistics(cv::Mat& frame) {
    int y_offset = 30;
    cv::Scalar text_color(255, 255, 255);
    cv::Scalar bg_color(0, 0, 0);
    
    // 绘制背景
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(400, 150), bg_color, -1);
    cv::rectangle(frame, cv::Point(10, 10), cv::Point(400, 150), text_color, 1);
    
    // 活跃对象数量
    auto active_objects = tracker_.getActiveObjects();
    std::string text = "Active Objects: " + std::to_string(active_objects.size());
    cv::putText(frame, text, cv::Point(20, y_offset), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
    y_offset += 25;
    
    // 区域统计
    auto zone_counts = counter_.getZoneCounts();
    for (const auto& pair : zone_counts) {
        text = pair.first + ": In=" + std::to_string(pair.second.first) + 
               " Out=" + std::to_string(pair.second.second);
        cv::putText(frame, text, cv::Point(20, y_offset), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        y_offset += 25;
    }
    
    // 线统计
    auto line_counts = counter_.getLineCounts();
    for (const auto& pair : line_counts) {
        text = pair.first + ": " + std::to_string(pair.second);
        cv::putText(frame, text, cv::Point(20, y_offset), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        y_offset += 25;
    }
}

void VideoProcessor::initializeClassNames() {
    class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
}

cv::Mat VideoProcessor::preprocessFrame(const cv::Mat& frame, LetterboxInfo& letterbox_info) {
    return preprocessLetterboxWithMeanStd(
        frame, input_width_, input_height_, 
        true, true, nullptr, nullptr,
        cv::Scalar(114, 114, 114), &letterbox_info
    );
}

std::vector<Detection> VideoProcessor::postprocessResults(const std::vector<float>& output,
                                                        const nvinfer1::Dims& output_dims,
                                                        const LetterboxInfo& letterbox_info,
                                                        int original_width,
                                                        int original_height) {
    std::vector<Detection> detections;
    
    // 解析输出维度
    int num_detections = 0;
    int num_classes = 0;
    bool layout_NC = true;
    
    if (output_dims.nbDims == 3) {
        // [B, N, C] 或 [B, C, N]
        int d1 = output_dims.d[1];
        int d2 = output_dims.d[2];
        if (d1 > d2) {
            num_detections = d1;
            num_classes = d2 - 4; // 减去坐标
            layout_NC = true;
        } else {
            num_detections = d2;
            num_classes = d1 - 4;
            layout_NC = false;
        }
    } else if (output_dims.nbDims == 2) {
        // [N, C] 或 [C, N]
        int d0 = output_dims.d[0];
        int d1 = output_dims.d[1];
        if (d0 > d1) {
            num_detections = d0;
            num_classes = d1 - 4;
            layout_NC = true;
        } else {
            num_detections = d1;
            num_classes = d0 - 4;
            layout_NC = false;
        }
    }
    
    if (num_detections <= 0 || num_classes <= 0) {
        return detections;
    }
    
    // 解码检测结果
    for (int i = 0; i < num_detections; ++i) {
        const float* det_data;
        if (layout_NC) {
            det_data = output.data() + i * (num_classes + 4);
        } else {
            // 简化处理，假设是NC布局
            continue;
        }
        
        float cx = det_data[0];
        float cy = det_data[1];
        float w = det_data[2];
        float h = det_data[3];
        
        // 找到最大置信度的类别
        float max_conf = 0.0f;
        int best_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            float conf = det_data[4 + c];
            if (conf > max_conf) {
                max_conf = conf;
                best_class = c;
            }
        }
        
        if (max_conf < conf_threshold_) {
            continue;
        }
        
        // 转换坐标：中心点格式到左上角格式
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        
        // 反向letterbox变换
        x1 = (x1 - letterbox_info.padX) / letterbox_info.scale;
        y1 = (y1 - letterbox_info.padY) / letterbox_info.scale;
        w = w / letterbox_info.scale;
        h = h / letterbox_info.scale;
        
        // 确保坐标在图像范围内
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_height)));
        w = std::min(w, static_cast<float>(original_width) - x1);
        h = std::min(h, static_cast<float>(original_height) - y1);
        
        if (w > 0 && h > 0) {
            detections.emplace_back(cv::Rect2f(x1, y1, w, h), best_class, max_conf);
        }
    }
    
    return detections;
}
