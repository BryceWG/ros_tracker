#ifndef VISUAL_TRACKER_HPP
#define VISUAL_TRACKER_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "kcftracker.hpp"

class VisualTracker {
public:
    VisualTracker();
    ~VisualTracker();

private:
    void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthCallback(const sensor_msgs::ImageConstPtr& msg);
    void onMouse(int event, int x, int y, int flags);
    static void onMouseWrapper(int event, int x, int y, int flags, void* userdata);
    void calculateCommand();
    void showDebugInfo(cv::Mat& display_image);

    // ROS相关
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber rgb_sub_;
    image_transport::Subscriber depth_sub_;
    ros::Publisher cmd_vel_pub_;

    // 图像处理相关
    cv::Mat rgb_image_;
    cv::Mat depth_image_;
    cv::Rect select_rect_;
    cv::Point origin_;
    cv::Rect track_rect_;

    // 跟踪器相关
    bool select_flag_;
    bool begin_track_;
    bool renew_roi_;
    KCFTracker* tracker_;

    // 参数
    static constexpr double MAX_LINEAR_SPEED = 0.5;
    static constexpr double MIN_LINEAR_SPEED = 0.0;
    static constexpr double MIN_DISTANCE = 2500;  // mm
    static constexpr double MAX_DISTANCE = 4000;  // mm
    static constexpr double MAX_ROTATION_SPEED = 0.75;
    static constexpr double K_ROTATION_SPEED = 0.01;

    // 运动控制
    double linear_speed_;
    double rotation_speed_;
    bool target_lost_;
};

#endif // VISUAL_TRACKER_HPP 