#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct TrackingConfig {
    // 模型相关配置
    std::string engine_path = "models/yolov8n.trt";
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.5f;
    
    // 追踪器配置
    float tracker_iou_threshold = 0.5f;
    int max_missed_frames = 10;
    int max_trajectory_length = 30;
    
    // 计数区域配置
    struct CountingZoneConfig {
        std::string name;
        std::vector<cv::Point> polygon;
        cv::Scalar color;
    };
    
    struct CrossingLineConfig {
        std::string name;
        cv::Point start;
        cv::Point end;
        cv::Scalar color;
    };
    
    std::vector<CountingZoneConfig> counting_zones;
    std::vector<CrossingLineConfig> crossing_lines;
    
    // 显示配置
    bool show_detection_boxes = true;
    bool show_tracking_boxes = true;
    bool show_trajectories = true;
    bool show_statistics = true;
    bool show_zone_overlay = true;
    
    // 输出配置
    bool save_video = false;
    std::string output_path = "output.mp4";
    int output_fps = 30;
    
    // 默认配置
    static TrackingConfig getDefaultConfig() {
        TrackingConfig config;
        
        // 添加默认计数区域
        CountingZoneConfig zone1;
        zone1.name = "Zone1";
        zone1.polygon = {
            cv::Point(100, 100),
            cv::Point(500, 100),
            cv::Point(500, 400),
            cv::Point(100, 400)
        };
        zone1.color = cv::Scalar(0, 255, 0);
        config.counting_zones.push_back(zone1);
        
        // 添加默认计数线
        CrossingLineConfig line1;
        line1.name = "Line1";
        line1.start = cv::Point(300, 50);
        line1.end = cv::Point(300, 600);
        line1.color = cv::Scalar(255, 0, 0);
        config.crossing_lines.push_back(line1);
        
        return config;
    }
    
    // 从文件加载配置
    bool loadFromFile(const std::string& config_file);
    
    // 保存配置到文件
    bool saveToFile(const std::string& config_file) const;
};
