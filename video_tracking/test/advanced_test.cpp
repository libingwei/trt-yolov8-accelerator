#include "../include/object_tracker.h"
#include "../include/zone_counter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

// 模拟检测结果生成器
class MockDetectionGenerator {
public:
    MockDetectionGenerator() : frame_count_(0) {}
    
    std::vector<Detection> generateDetections(int frame_num) {
        std::vector<Detection> detections;
        
        // 模拟移动的人员
        float x1 = 50 + (frame_num * 2) % 400;  // 从左到右移动
        float y1 = 150 + sin(frame_num * 0.1) * 50;  // 上下摆动
        detections.push_back(Detection(cv::Rect2f(x1, y1, 60, 120), 0, 0.85f)); // person
        
        // 模拟另一个移动的对象
        float x2 = 300 - (frame_num * 1.5) % 300;  // 从右到左移动
        float y2 = 200;
        detections.push_back(Detection(cv::Rect2f(x2, y2, 80, 60), 2, 0.9f)); // car
        
        // 偶尔添加新对象
        if (frame_num % 50 == 0) {
            float x3 = rand() % 400;
            float y3 = rand() % 300 + 100;
            detections.push_back(Detection(cv::Rect2f(x3, y3, 50, 100), 0, 0.7f)); // person
        }
        
        return detections;
    }
    
private:
    int frame_count_;
};

void testBasicTracking() {
    std::cout << "=== 测试基本追踪功能 ===" << std::endl;
    
    ObjectTracker tracker(0.5f, 10);
    MockDetectionGenerator generator;
    
    for (int frame = 0; frame < 100; ++frame) {
        auto detections = generator.generateDetections(frame);
        tracker.update(detections);
        
        auto active_objects = tracker.getActiveObjects();
        
        if (frame % 10 == 0) {
            std::cout << "Frame " << frame << ": " 
                     << detections.size() << " detections, "
                     << active_objects.size() << " tracked objects" << std::endl;
            
            for (const auto& obj : active_objects) {
                std::cout << "  Object ID " << obj.id 
                         << ", Class " << obj.class_id
                         << ", Position (" << obj.center.x << ", " << obj.center.y << ")"
                         << ", Trajectory length: " << obj.trajectory.size() << std::endl;
            }
        }
    }
    
    std::cout << "基本追踪测试完成" << std::endl << std::endl;
}

void testZoneCounting() {
    std::cout << "=== 测试区域计数功能 ===" << std::endl;
    
    ObjectTracker tracker(0.5f, 10);
    ZoneCounter counter;
    
    // 设置计数区域
    std::vector<cv::Point> zone1 = {
        cv::Point(150, 120),
        cv::Point(300, 120),
        cv::Point(300, 250),
        cv::Point(150, 250)
    };
    counter.addCountingZone(CountingZone(zone1, "TestZone", cv::Scalar(0, 255, 0)));
    
    // 设置计数线
    counter.addCrossingLine(CrossingLine(cv::Point(250, 50), cv::Point(250, 350), 
                                       "TestLine", cv::Scalar(255, 0, 0)));
    
    MockDetectionGenerator generator;
    
    for (int frame = 0; frame < 150; ++frame) {
        auto detections = generator.generateDetections(frame);
        tracker.update(detections);
        
        auto tracked_objects = tracker.getActiveObjects();
        counter.updateCounts(tracked_objects);
        
        if (frame % 30 == 0) {
            std::cout << "Frame " << frame << ":" << std::endl;
            
            auto zone_counts = counter.getZoneCounts();
            for (const auto& pair : zone_counts) {
                std::cout << "  Zone " << pair.first 
                         << ": Enter=" << pair.second.first 
                         << ", Exit=" << pair.second.second << std::endl;
            }
            
            auto line_counts = counter.getLineCounts();
            for (const auto& pair : line_counts) {
                std::cout << "  Line " << pair.first 
                         << ": Crossings=" << pair.second << std::endl;
            }
        }
    }
    
    std::cout << "区域计数测试完成" << std::endl << std::endl;
}

void testVisualTracking() {
    std::cout << "=== 测试可视化追踪 ===" << std::endl;
    std::cout << "将打开窗口显示追踪效果，按 'q' 退出，按 'r' 重置" << std::endl;
    
    ObjectTracker tracker(0.5f, 10);
    ZoneCounter counter;
    MockDetectionGenerator generator;
    
    // 设置计数区域和线
    std::vector<cv::Point> zone1 = {
        cv::Point(150, 120),
        cv::Point(350, 120),
        cv::Point(350, 280),
        cv::Point(150, 280)
    };
    counter.addCountingZone(CountingZone(zone1, "Zone1", cv::Scalar(0, 255, 0)));
    counter.addCrossingLine(CrossingLine(cv::Point(300, 50), cv::Point(300, 350), 
                                       "Line1", cv::Scalar(255, 0, 0)));
    
    // COCO类别名称
    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"
    };
    
    cv::namedWindow("Tracking Test", cv::WINDOW_AUTOSIZE);
    
    for (int frame = 0; frame < 500; ++frame) {
        // 创建画布
        cv::Mat canvas(400, 640, CV_8UC3, cv::Scalar(50, 50, 50));
        
        // 生成检测结果
        auto detections = generator.generateDetections(frame);
        tracker.update(detections);
        
        auto tracked_objects = tracker.getActiveObjects();
        counter.updateCounts(tracked_objects);
        
        // 绘制区域和线
        counter.drawZones(canvas);
        
        // 绘制检测框（黄色）
        for (const auto& det : detections) {
            cv::rectangle(canvas, det.bbox, cv::Scalar(0, 255, 255), 2);
        }
        
        // 绘制追踪对象
        for (const auto& obj : tracked_objects) {
            if (!obj.is_active) continue;
            
            // 绘制边界框（绿色）
            cv::Scalar color(0, 255, 0);
            cv::rectangle(canvas, obj.bbox, color, 2);
            
            // 绘制ID和类别
            std::string label = "ID:" + std::to_string(obj.id);
            if (obj.class_id < class_names.size()) {
                label += " " + class_names[obj.class_id];
            }
            
            cv::Point label_pos(obj.bbox.x, obj.bbox.y - 10);
            cv::putText(canvas, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            
            // 绘制轨迹（红色）
            if (obj.trajectory.size() > 1) {
                for (size_t i = 1; i < obj.trajectory.size(); ++i) {
                    cv::line(canvas, obj.trajectory[i-1], obj.trajectory[i], 
                            cv::Scalar(0, 0, 255), 2);
                }
            }
            
            // 绘制中心点
            cv::circle(canvas, obj.center, 3, cv::Scalar(0, 0, 255), -1);
        }
        
        // 绘制统计信息
        int y_offset = 30;
        cv::Scalar text_color(255, 255, 255);
        cv::Scalar bg_color(0, 0, 0);
        
        // 背景
        cv::rectangle(canvas, cv::Point(10, 10), cv::Point(300, 120), bg_color, -1);
        cv::rectangle(canvas, cv::Point(10, 10), cv::Point(300, 120), text_color, 1);
        
        // 帧数和对象数
        std::string text = "Frame: " + std::to_string(frame);
        cv::putText(canvas, text, cv::Point(20, y_offset), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
        y_offset += 20;
        
        text = "Objects: " + std::to_string(tracked_objects.size());
        cv::putText(canvas, text, cv::Point(20, y_offset), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
        y_offset += 20;
        
        // 区域统计
        auto zone_counts = counter.getZoneCounts();
        for (const auto& pair : zone_counts) {
            text = pair.first + ": " + std::to_string(pair.second.first) + "/" + 
                   std::to_string(pair.second.second);
            cv::putText(canvas, text, cv::Point(20, y_offset), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
            y_offset += 20;
        }
        
        cv::imshow("Tracking Test", canvas);
        
        char key = cv::waitKey(50) & 0xFF;
        if (key == 'q') {
            break;
        } else if (key == 'r') {
            tracker.reset();
            counter.resetCounts();
            std::cout << "追踪器和计数器已重置" << std::endl;
        }
    }
    
    cv::destroyAllWindows();
    std::cout << "可视化测试完成" << std::endl << std::endl;
}

void testPerformance() {
    std::cout << "=== 性能测试 ===" << std::endl;
    
    ObjectTracker tracker(0.5f, 10);
    ZoneCounter counter;
    MockDetectionGenerator generator;
    
    // 设置区域
    std::vector<cv::Point> zone1 = {
        cv::Point(100, 100), cv::Point(300, 100),
        cv::Point(300, 300), cv::Point(100, 300)
    };
    counter.addCountingZone(CountingZone(zone1, "PerfZone"));
    
    const int num_frames = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int frame = 0; frame < num_frames; ++frame) {
        auto detections = generator.generateDetections(frame);
        tracker.update(detections);
        auto tracked_objects = tracker.getActiveObjects();
        counter.updateCounts(tracked_objects);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double fps = num_frames * 1000.0 / duration.count();
    
    std::cout << "处理 " << num_frames << " 帧" << std::endl;
    std::cout << "总时间: " << duration.count() << " ms" << std::endl;
    std::cout << "平均FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
    std::cout << "性能测试完成" << std::endl << std::endl;
}

int main() {
    std::cout << "YOLOv8 视频追踪系统 - 扩展测试程序" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    try {
        // 运行各项测试
        testBasicTracking();
        testZoneCounting();
        testPerformance();
        
        // 询问是否运行可视化测试
        std::cout << "是否运行可视化测试？这将打开一个窗口显示追踪效果。(y/n): ";
        char choice;
        std::cin >> choice;
        
        if (choice == 'y' || choice == 'Y') {
            testVisualTracking();
        }
        
        std::cout << "所有测试完成！" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
}
