#include "../include/video_processor.h"
#include <iostream>
#include <fstream>
#include <iomanip>

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

// 其他方法的简化实现...
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
