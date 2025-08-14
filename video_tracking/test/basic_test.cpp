#include "../include/object_tracker.h"
#include "../include/zone_counter.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "Basic Object Tracking Test\n";
    
    // 测试追踪器
    ObjectTracker tracker(0.5f, 10);
    
    // 创建一些模拟检测结果
    std::vector<Detection> detections1 = {
        Detection(cv::Rect2f(100, 100, 50, 100), 0, 0.8f),  // person
        Detection(cv::Rect2f(200, 150, 80, 60), 2, 0.9f)    // car
    };
    
    std::vector<Detection> detections2 = {
        Detection(cv::Rect2f(105, 105, 50, 100), 0, 0.85f), // person moved slightly
        Detection(cv::Rect2f(250, 160, 80, 60), 2, 0.87f)   // car moved
    };
    
    // 第一帧
    tracker.update(detections1);
    auto objects1 = tracker.getActiveObjects();
    std::cout << "Frame 1: " << objects1.size() << " tracked objects\n";
    
    // 第二帧
    tracker.update(detections2);
    auto objects2 = tracker.getActiveObjects();
    std::cout << "Frame 2: " << objects2.size() << " tracked objects\n";
    
    for (const auto& obj : objects2) {
        std::cout << "Object ID: " << obj.id 
                  << ", Class: " << obj.class_id 
                  << ", Position: (" << obj.center.x << ", " << obj.center.y << ")\n";
    }
    
    // 测试区域计数器
    std::cout << "\nTesting Zone Counter:\n";
    ZoneCounter counter;
    
    // 创建一个测试区域
    std::vector<cv::Point> zone_polygon = {
        cv::Point(150, 120),
        cv::Point(300, 120),
        cv::Point(300, 250),
        cv::Point(150, 250)
    };
    counter.addCountingZone(CountingZone(zone_polygon, "TestZone"));
    
    // 更新计数
    counter.updateCounts(objects2);
    
    auto zone_counts = counter.getZoneCounts();
    for (const auto& pair : zone_counts) {
        std::cout << "Zone " << pair.first 
                  << ": Enter=" << pair.second.first 
                  << ", Exit=" << pair.second.second << "\n";
    }
    
    std::cout << "Basic test completed successfully!\n";
    return 0;
}
