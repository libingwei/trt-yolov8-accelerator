#include "../include/tracking_config.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool TrackingConfig::loadFromFile(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "无法打开配置文件: " << config_file << std::endl;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释和空行
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            // 去除空格
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            // 解析配置项
            if (key == "engine_path") {
                engine_path = value;
            } else if (key == "input_width") {
                input_width = std::stoi(value);
            } else if (key == "input_height") {
                input_height = std::stoi(value);
            } else if (key == "conf_threshold") {
                conf_threshold = std::stof(value);
            } else if (key == "iou_threshold") {
                iou_threshold = std::stof(value);
            } else if (key == "tracker_iou_threshold") {
                tracker_iou_threshold = std::stof(value);
            } else if (key == "max_missed_frames") {
                max_missed_frames = std::stoi(value);
            } else if (key == "show_detection_boxes") {
                show_detection_boxes = (value == "true" || value == "1");
            } else if (key == "show_tracking_boxes") {
                show_tracking_boxes = (value == "true" || value == "1");
            } else if (key == "show_trajectories") {
                show_trajectories = (value == "true" || value == "1");
            } else if (key == "output_path") {
                output_path = value;
            }
        }
    }
    
    return true;
}

bool TrackingConfig::saveToFile(const std::string& config_file) const {
    std::ofstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "无法创建配置文件: " << config_file << std::endl;
        return false;
    }
    
    file << "# YOLOv8 视频追踪配置文件\n";
    file << "# 模型配置\n";
    file << "engine_path=" << engine_path << "\n";
    file << "input_width=" << input_width << "\n";
    file << "input_height=" << input_height << "\n";
    file << "conf_threshold=" << conf_threshold << "\n";
    file << "iou_threshold=" << iou_threshold << "\n";
    file << "\n# 追踪器配置\n";
    file << "tracker_iou_threshold=" << tracker_iou_threshold << "\n";
    file << "max_missed_frames=" << max_missed_frames << "\n";
    file << "\n# 显示配置\n";
    file << "show_detection_boxes=" << (show_detection_boxes ? "true" : "false") << "\n";
    file << "show_tracking_boxes=" << (show_tracking_boxes ? "true" : "false") << "\n";
    file << "show_trajectories=" << (show_trajectories ? "true" : "false") << "\n";
    file << "\n# 输出配置\n";
    file << "output_path=" << output_path << "\n";
    
    return true;
}
