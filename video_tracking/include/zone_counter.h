#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include "object_tracker.h"

struct CountingZone {
    std::vector<cv::Point> polygon;
    std::string name;
    int enter_count;
    int exit_count;
    cv::Scalar color;
    
    CountingZone(const std::vector<cv::Point>& _polygon, const std::string& _name, 
                 const cv::Scalar& _color = cv::Scalar(0, 255, 0))
        : polygon(_polygon), name(_name), enter_count(0), exit_count(0), color(_color) {}
    
    bool contains(const cv::Point2f& point) const;
};

struct CrossingLine {
    cv::Point start;
    cv::Point end;
    std::string name;
    int cross_count;
    cv::Scalar color;
    
    CrossingLine(const cv::Point& _start, const cv::Point& _end, const std::string& _name,
                 const cv::Scalar& _color = cv::Scalar(255, 0, 0))
        : start(_start), end(_end), name(_name), cross_count(0), color(_color) {}
};

class ZoneCounter {
public:
    ZoneCounter() = default;
    ~ZoneCounter() = default;
    
    // 添加计数区域
    void addCountingZone(const CountingZone& zone);
    
    // 添加计数线
    void addCrossingLine(const CrossingLine& line);
    
    // 更新计数（基于追踪对象）
    void updateCounts(const std::vector<TrackedObject>& tracked_objects);
    
    // 绘制区域和统计信息
    void drawZones(cv::Mat& frame) const;
    
    // 获取区域统计
    std::map<std::string, std::pair<int, int>> getZoneCounts() const;
    
    // 获取线统计
    std::map<std::string, int> getLineCounts() const;
    
    // 重置计数
    void resetCounts();
    
private:
    std::vector<CountingZone> counting_zones_;
    std::vector<CrossingLine> crossing_lines_;
    std::map<int, std::vector<cv::Point2f>> object_history_; // 对象ID -> 历史位置
    std::map<int, std::string> last_zone_; // 对象ID -> 最后所在区域
    
    // 检查两条线是否相交
    bool linesIntersect(const cv::Point2f& p1, const cv::Point2f& q1,
                       const cv::Point2f& p2, const cv::Point2f& q2) const;
    
    // 检查点是否在多边形内
    bool pointInPolygon(const cv::Point2f& point, const std::vector<cv::Point>& polygon) const;
};
