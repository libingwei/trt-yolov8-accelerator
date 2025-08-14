#include "include/video_processor.h"
#include <iostream>
#include <string>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <engine.trt> [options]\n"
              << "Options:\n"
              << "  --video <path>      Process video file\n"
              << "  --camera <id>       Process camera stream (default: 0)\n"
              << "  --output <path>     Output video path (optional)\n"
              << "  --width <w>         Input width (default: 640)\n"
              << "  --height <h>        Input height (default: 640)\n"
              << "  --conf <threshold>  Confidence threshold (default: 0.25)\n"
              << "  --iou <threshold>   IoU threshold (default: 0.5)\n"
              << "  --no-preview        Disable preview window\n"
              << "  --help              Show this help\n"
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
    int width = 640;
    int height = 640;
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
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--iou" && i + 1 < argc) {
            iou_threshold = std::stof(argv[++i]);
        } else if (arg == "--no-preview") {
            show_preview = false;
        }
    }
    
    // 验证输入
    if (!use_camera && video_path.empty()) {
        std::cerr << "Error: Must specify either --video or --camera\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // 创建视频处理器
    VideoProcessor processor(engine_path, width, height, conf_threshold, iou_threshold);
    
    // 初始化
    if (!processor.initialize()) {
        std::cerr << "Failed to initialize video processor\n";
        return 1;
    }
    
    // 设置计数区域（示例）
    processor.setupCountingZones();
    
    std::cout << "Starting video processing...\n";
    std::cout << "Engine: " << engine_path << "\n";
    std::cout << "Input size: " << width << "x" << height << "\n";
    std::cout << "Confidence threshold: " << conf_threshold << "\n";
    std::cout << "IoU threshold: " << iou_threshold << "\n";
    
    bool success = false;
    if (use_camera) {
        std::cout << "Processing camera stream (ID: " << camera_id << ")\n";
        success = processor.processCameraStream(camera_id, output_path);
    } else {
        std::cout << "Processing video: " << video_path << "\n";
        success = processor.processVideo(video_path, output_path, show_preview);
    }
    
    if (success) {
        std::cout << "Video processing completed successfully!\n";
        if (!output_path.empty()) {
            std::cout << "Output saved to: " << output_path << "\n";
        }
    } else {
        std::cerr << "Video processing failed!\n";
        return 1;
    }
    
    return 0;
}
