#include "include/video_processor.h"
#include <iostream>
#include <string>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <engine_path> [options]\n"
              << "Options:\n"
              << "  --video <path>        Process video file\n"
              << "  --camera <id>         Process camera stream (default: 0)\n"
              << "  --output <path>       Save output video\n"
              << "  --width <width>       Input width (default: 640)\n"
              << "  --height <height>     Input height (default: 640)\n"
              << "  --conf <threshold>    Confidence threshold (default: 0.25)\n"
              << "  --iou <threshold>     IoU threshold (default: 0.5)\n"
              << "  --no-preview          Disable preview window\n"
              << "  --help                Show this help\n"
              << "\nExamples:\n"
              << "  " << program_name << " models/yolov8n.trt --video test.mp4\n"
              << "  " << program_name << " models/yolov8n.trt --camera 0 --output output.mp4\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string engine_path = argv[1];
    std::string video_path;
    std::string output_path;
    int camera_id = 0;
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.5f;
    bool use_camera = false;
    bool show_preview = true;
    
    // 解析命令行参数
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
            use_camera = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            input_width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            input_height = std::stoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--iou" && i + 1 < argc) {
            iou_threshold = std::stof(argv[++i]);
        } else if (arg == "--no-preview") {
            show_preview = false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // 检查输入参数
    if (!use_camera && video_path.empty()) {
        std::cerr << "Error: Please specify either --video or --camera" << std::endl;
        return 1;
    }
    
    std::cout << "Initializing Video Tracker..." << std::endl;
    std::cout << "Engine: " << engine_path << std::endl;
    std::cout << "Input size: " << input_width << "x" << input_height << std::endl;
    std::cout << "Confidence threshold: " << conf_threshold << std::endl;
    std::cout << "IoU threshold: " << iou_threshold << std::endl;
    
    // 创建视频处理器
    VideoProcessor processor(engine_path, input_width, input_height, conf_threshold, iou_threshold);
    
    // 初始化
    if (!processor.initialize()) {
        std::cerr << "Failed to initialize video processor" << std::endl;
        return 1;
    }
    
    // 设置计数区域（可以根据需要修改）
    processor.setupCountingZones();
    
    // 处理视频或摄像头
    bool success = false;
    if (use_camera) {
        std::cout << "Starting camera stream..." << std::endl;
        success = processor.processCameraStream(camera_id, output_path);
    } else {
        std::cout << "Processing video: " << video_path << std::endl;
        success = processor.processVideo(video_path, output_path, show_preview);
    }
    
    if (success) {
        std::cout << "Processing completed successfully!" << std::endl;
        
        // 打印最终统计
        auto zone_counts = processor.getCounter().getZoneCounts();
        auto line_counts = processor.getCounter().getLineCounts();
        
        std::cout << "\n=== Final Statistics ===" << std::endl;
        for (const auto& zone : zone_counts) {
            std::cout << zone.first << " - Enter: " << zone.second.first 
                      << ", Exit: " << zone.second.second << std::endl;
        }
        for (const auto& line : line_counts) {
            std::cout << line.first << " - Crossings: " << line.second << std::endl;
        }
    } else {
        std::cerr << "Processing failed!" << std::endl;
        return 1;
    }
    
    return 0;
}
