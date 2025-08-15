#pragma once

#include <trt_utils/trt_runtime.h>
#include <trt_utils/trt_common.h>
#include <trt_utils/trt_preprocess.h>
#include <trt_utils/trt_vision.h>
#include <trt_utils/trt_decode.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>
#include "object_tracker.h"
#include "zone_counter.h"

class VideoProcessor {
public:
    VideoProcessor(const std::string& engine_path, 
                   int input_width = 640, 
                   int input_height = 640,
                   float conf_threshold = 0.25f,
                   float iou_threshold = 0.5f);
    ~VideoProcessor() = default;
    
    // 初始化处理器
    bool initialize();
    
    // 处理单帧
    std::vector<Detection> processFrame(const cv::Mat& frame);
    
    // 处理视频文件
    bool processVideo(const std::string& video_path, 
                     const std::string& output_path = "",
                     bool show_preview = true);
    
    // 处理摄像头输入
    bool processCameraStream(int camera_id = 0, 
                           const std::string& output_path = "");
    
    // 设置计数区域
    void setupCountingZones();
    
    // 获取追踪器
    ObjectTracker& getTracker() { return tracker_; }
    
    // 获取计数器
    ZoneCounter& getCounter() { return counter_; }
    
    // 绘制检测和追踪结果
    void drawResults(cv::Mat& frame, 
                    const std::vector<Detection>& detections,
                    const std::vector<TrackedObject>& tracked_objects);
    
    // 显示统计信息
    void drawStatistics(cv::Mat& frame);
    
private:
    std::string engine_path_;
    int input_width_;
    int input_height_;
    float conf_threshold_;
    float iou_threshold_;
    
    std::unique_ptr<TrtRunner> runner_;
    std::unique_ptr<nvinfer1::ILogger> logger_;
    
    ObjectTracker tracker_;
    ZoneCounter counter_;
    
    // COCO类别名称
    std::vector<std::string> class_names_;
    
    // 初始化类别名称
    void initializeClassNames();
    
    // 预处理帧
    cv::Mat preprocessFrame(const cv::Mat& frame, LetterboxInfo& letterbox_info);
    
    // 后处理结果
    std::vector<Detection> postprocessResults(const std::vector<float>& output,
                                            const nvinfer1::Dims& output_dims,
                                            const LetterboxInfo& letterbox_info,
                                            int original_width,
                                            int original_height);
};
