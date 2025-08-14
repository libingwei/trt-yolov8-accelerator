#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

struct TrackedObject {
    int id;
    int class_id;
    cv::Rect2f bbox;
    float confidence;
    cv::Point2f center;
    std::vector<cv::Point2f> trajectory;
    std::chrono::steady_clock::time_point last_seen;
    bool is_active;
    int missed_frames;
    
    TrackedObject(int _id, int _class_id, const cv::Rect2f& _bbox, float _conf)
        : id(_id), class_id(_class_id), bbox(_bbox), confidence(_conf),
          center(_bbox.x + _bbox.width/2, _bbox.y + _bbox.height/2),
          last_seen(std::chrono::steady_clock::now()), is_active(true), missed_frames(0) {
        trajectory.push_back(center);
    }
};

struct Detection {
    cv::Rect2f bbox;
    int class_id;
    float confidence;
    
    Detection(const cv::Rect2f& _bbox, int _class_id, float _conf)
        : bbox(_bbox), class_id(_class_id), confidence(_conf) {}
};

class ObjectTracker {
public:
    ObjectTracker(float iou_threshold = 0.5f, int max_missed_frames = 10);
    ~ObjectTracker() = default;
    
    // 更新追踪器，输入新的检测结果
    void update(const std::vector<Detection>& detections);
    
    // 获取当前活跃的追踪对象
    std::vector<TrackedObject> getActiveObjects() const;
    
    // 获取所有追踪对象（包括不活跃的）
    std::vector<TrackedObject> getAllObjects() const;
    
    // 清理不活跃的对象
    void cleanupInactiveObjects();
    
    // 重置追踪器
    void reset();
    
    // 获取下一个ID
    int getNextId() { return next_id_++; }
    
private:
    std::map<int, TrackedObject> tracked_objects_;
    int next_id_;
    float iou_threshold_;
    int max_missed_frames_;
    
    // 计算两个边界框的IoU
    float calculateIoU(const cv::Rect2f& box1, const cv::Rect2f& box2) const;
    
    // 匹配检测结果和追踪对象
    void matchDetectionsToTracks(const std::vector<Detection>& detections,
                                std::vector<std::pair<int, int>>& matches,
                                std::vector<int>& unmatched_detections,
                                std::vector<int>& unmatched_tracks);
};
