#include "../include/object_tracker.h"
#include <algorithm>
#include <unordered_set>

ObjectTracker::ObjectTracker(float iou_threshold, int max_missed_frames)
    : next_id_(1), iou_threshold_(iou_threshold), max_missed_frames_(max_missed_frames) {
}

void ObjectTracker::update(const std::vector<Detection>& detections) {
    // 匹配检测结果和现有追踪对象
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_detections;
    std::vector<int> unmatched_tracks;
    
    matchDetectionsToTracks(detections, matches, unmatched_detections, unmatched_tracks);
    
    // 更新匹配的追踪对象
    for (const auto& match : matches) {
        int track_id = match.first;
        int detection_idx = match.second;
        
        auto& obj = tracked_objects_[track_id];
        const auto& det = detections[detection_idx];
        
        // 更新边界框和置信度
        obj.bbox = det.bbox;
        obj.confidence = det.confidence;
        obj.center = cv::Point2f(det.bbox.x + det.bbox.width/2, det.bbox.y + det.bbox.height/2);
        obj.trajectory.push_back(obj.center);
        obj.last_seen = std::chrono::steady_clock::now();
        obj.is_active = true;
        obj.missed_frames = 0;
        
        // 限制轨迹长度
        if (obj.trajectory.size() > 30) {
            obj.trajectory.erase(obj.trajectory.begin());
        }
    }
    
    // 处理未匹配的追踪对象
    for (int track_id : unmatched_tracks) {
        auto& obj = tracked_objects_[track_id];
        obj.missed_frames++;
        
        if (obj.missed_frames > max_missed_frames_) {
            obj.is_active = false;
        }
    }
    
    // 为未匹配的检测结果创建新的追踪对象
    for (int det_idx : unmatched_detections) {
        const auto& det = detections[det_idx];
        int new_id = getNextId();
        tracked_objects_[new_id] = TrackedObject(new_id, det.class_id, det.bbox, det.confidence);
    }
    
    // 清理长时间不活跃的对象
    cleanupInactiveObjects();
}

std::vector<TrackedObject> ObjectTracker::getActiveObjects() const {
    std::vector<TrackedObject> active_objects;
    for (const auto& pair : tracked_objects_) {
        if (pair.second.is_active) {
            active_objects.push_back(pair.second);
        }
    }
    return active_objects;
}

std::vector<TrackedObject> ObjectTracker::getAllObjects() const {
    std::vector<TrackedObject> all_objects;
    for (const auto& pair : tracked_objects_) {
        all_objects.push_back(pair.second);
    }
    return all_objects;
}

void ObjectTracker::cleanupInactiveObjects() {
    auto it = tracked_objects_.begin();
    while (it != tracked_objects_.end()) {
        if (!it->second.is_active) {
            auto current_time = std::chrono::steady_clock::now();
            auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(
                current_time - it->second.last_seen).count();
            
            // 删除超过30秒未见的对象
            if (time_diff > 30) {
                it = tracked_objects_.erase(it);
            } else {
                ++it;
            }
        } else {
            ++it;
        }
    }
}

void ObjectTracker::reset() {
    tracked_objects_.clear();
    next_id_ = 1;
}

float ObjectTracker::calculateIoU(const cv::Rect2f& box1, const cv::Rect2f& box2) const {
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;
    
    return union_area > 0 ? intersection / union_area : 0.0f;
}

void ObjectTracker::matchDetectionsToTracks(const std::vector<Detection>& detections,
                                           std::vector<std::pair<int, int>>& matches,
                                           std::vector<int>& unmatched_detections,
                                           std::vector<int>& unmatched_tracks) {
    matches.clear();
    unmatched_detections.clear();
    unmatched_tracks.clear();
    
    if (detections.empty() || tracked_objects_.empty()) {
        for (size_t i = 0; i < detections.size(); ++i) {
            unmatched_detections.push_back(i);
        }
        for (const auto& pair : tracked_objects_) {
            if (pair.second.is_active) {
                unmatched_tracks.push_back(pair.first);
            }
        }
        return;
    }
    
    // 计算IoU矩阵
    std::vector<std::vector<float>> iou_matrix;
    std::vector<int> active_track_ids;
    
    for (const auto& pair : tracked_objects_) {
        if (pair.second.is_active) {
            active_track_ids.push_back(pair.first);
        }
    }
    
    iou_matrix.resize(active_track_ids.size());
    for (size_t i = 0; i < active_track_ids.size(); ++i) {
        iou_matrix[i].resize(detections.size());
        const auto& track_bbox = tracked_objects_[active_track_ids[i]].bbox;
        
        for (size_t j = 0; j < detections.size(); ++j) {
            // 只匹配相同类别的对象
            if (tracked_objects_[active_track_ids[i]].class_id == detections[j].class_id) {
                iou_matrix[i][j] = calculateIoU(track_bbox, detections[j].bbox);
            } else {
                iou_matrix[i][j] = 0.0f;
            }
        }
    }
    
    // 贪心匹配算法
    std::unordered_set<int> matched_tracks;
    std::unordered_set<int> matched_detections;
    
    for (size_t i = 0; i < active_track_ids.size(); ++i) {
        if (matched_tracks.count(i)) continue;
        
        float max_iou = 0.0f;
        int best_detection = -1;
        
        for (size_t j = 0; j < detections.size(); ++j) {
            if (matched_detections.count(j)) continue;
            
            if (iou_matrix[i][j] > max_iou && iou_matrix[i][j] > iou_threshold_) {
                max_iou = iou_matrix[i][j];
                best_detection = j;
            }
        }
        
        if (best_detection != -1) {
            matches.push_back({active_track_ids[i], best_detection});
            matched_tracks.insert(i);
            matched_detections.insert(best_detection);
        }
    }
    
    // 收集未匹配的追踪对象和检测结果
    for (size_t i = 0; i < active_track_ids.size(); ++i) {
        if (!matched_tracks.count(i)) {
            unmatched_tracks.push_back(active_track_ids[i]);
        }
    }
    
    for (size_t j = 0; j < detections.size(); ++j) {
        if (!matched_detections.count(j)) {
            unmatched_detections.push_back(j);
        }
    }
}
