#include "../include/zone_counter.h"
#include <opencv2/imgproc.hpp>

bool CountingZone::contains(const cv::Point2f& point) const {
    return cv::pointPolygonTest(polygon, point, false) >= 0;
}

void ZoneCounter::addCountingZone(const CountingZone& zone) {
    counting_zones_.push_back(zone);
}

void ZoneCounter::addCrossingLine(const CrossingLine& line) {
    crossing_lines_.push_back(line);
}

void ZoneCounter::updateCounts(const std::vector<TrackedObject>& tracked_objects) {
    for (const auto& obj : tracked_objects) {
        if (!obj.is_active) continue;
        
        int obj_id = obj.id;
        cv::Point2f current_pos = obj.center;
        
        // 更新对象历史位置
        object_history_[obj_id].push_back(current_pos);
        if (object_history_[obj_id].size() > 10) {
            object_history_[obj_id].erase(object_history_[obj_id].begin());
        }
        
        // 检查区域进出
        for (auto& zone : counting_zones_) {
            bool is_in_zone = zone.contains(current_pos);
            std::string zone_name = zone.name;
            
            if (last_zone_.count(obj_id)) {
                std::string last_zone_name = last_zone_[obj_id];
                
                if (last_zone_name != zone_name && is_in_zone) {
                    // 进入新区域
                    zone.enter_count++;
                    last_zone_[obj_id] = zone_name;
                } else if (last_zone_name == zone_name && !is_in_zone) {
                    // 离开当前区域
                    zone.exit_count++;
                    last_zone_[obj_id] = "";
                }
            } else if (is_in_zone) {
                // 首次进入区域
                zone.enter_count++;
                last_zone_[obj_id] = zone_name;
            }
        }
        
        // 检查线段穿越
        if (object_history_[obj_id].size() >= 2) {
            cv::Point2f prev_pos = object_history_[obj_id][object_history_[obj_id].size() - 2];
            
            for (auto& line : crossing_lines_) {
                cv::Point2f line_start(line.start.x, line.start.y);
                cv::Point2f line_end(line.end.x, line.end.y);
                
                if (linesIntersect(prev_pos, current_pos, line_start, line_end)) {
                    line.cross_count++;
                }
            }
        }
    }
}

void ZoneCounter::drawZones(cv::Mat& frame) const {
    // 绘制计数区域
    for (const auto& zone : counting_zones_) {
        // 绘制多边形
        std::vector<cv::Point> pts = zone.polygon;
        const cv::Point* points = pts.data();
        int npts = pts.size();
        cv::polylines(frame, &points, &npts, 1, true, zone.color, 2);
        
        // 填充半透明区域
        cv::Mat overlay;
        frame.copyTo(overlay);
        cv::fillPoly(overlay, &points, &npts, 1, zone.color);
        cv::addWeighted(frame, 0.7, overlay, 0.3, 0, frame);
        
        // 显示区域名称和计数
        cv::Point text_pos = zone.polygon[0];
        text_pos.y -= 10;
        std::string text = zone.name + ": In=" + std::to_string(zone.enter_count) + 
                          " Out=" + std::to_string(zone.exit_count);
        cv::putText(frame, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, zone.color, 2);
    }
    
    // 绘制计数线
    for (const auto& line : crossing_lines_) {
        cv::line(frame, line.start, line.end, line.color, 3);
        
        // 显示线名称和计数
        cv::Point text_pos = line.start;
        text_pos.y -= 10;
        std::string text = line.name + ": " + std::to_string(line.cross_count);
        cv::putText(frame, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, line.color, 2);
    }
}

std::map<std::string, std::pair<int, int>> ZoneCounter::getZoneCounts() const {
    std::map<std::string, std::pair<int, int>> counts;
    for (const auto& zone : counting_zones_) {
        counts[zone.name] = {zone.enter_count, zone.exit_count};
    }
    return counts;
}

std::map<std::string, int> ZoneCounter::getLineCounts() const {
    std::map<std::string, int> counts;
    for (const auto& line : crossing_lines_) {
        counts[line.name] = line.cross_count;
    }
    return counts;
}

void ZoneCounter::resetCounts() {
    for (auto& zone : counting_zones_) {
        zone.enter_count = 0;
        zone.exit_count = 0;
    }
    for (auto& line : crossing_lines_) {
        line.cross_count = 0;
    }
    object_history_.clear();
    last_zone_.clear();
}

bool ZoneCounter::linesIntersect(const cv::Point2f& p1, const cv::Point2f& q1,
                                const cv::Point2f& p2, const cv::Point2f& q2) const {
    auto orientation = [](const cv::Point2f& p, const cv::Point2f& q, const cv::Point2f& r) {
        float val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if (abs(val) < 1e-9) return 0;
        return (val > 0) ? 1 : 2;
    };
    
    auto onSegment = [](const cv::Point2f& p, const cv::Point2f& q, const cv::Point2f& r) {
        return q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
               q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y);
    };
    
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);
    
    // 一般情况
    if (o1 != o2 && o3 != o4) return true;
    
    // 特殊情况
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;
    
    return false;
}

bool ZoneCounter::pointInPolygon(const cv::Point2f& point, const std::vector<cv::Point>& polygon) const {
    return cv::pointPolygonTest(polygon, point, false) >= 0;
}
