#ifndef VISUAL_TRACKER_HPP
#define VISUAL_TRACKER_HPP

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Twist.h>
#include <opencv2/opencv.hpp>
#include "kcftracker.hpp"

// 定义常量
#define MAX_LINEAR_SPEED 0.5
#define MIN_LINEAR_SPEED 0.0
#define MIN_DISTANCE 2.5  // 米
#define MAX_DISTANCE 4.0  // 米
#define MAX_ROTATION_SPEED 0.75
#define K_ROTATION_SPEED 0.01

class VisualTracker
{
public:
    VisualTracker();
    ~VisualTracker();

private:
    // ROS相关
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber rgb_sub_;
    image_transport::Subscriber depth_sub_;
    ros::Publisher cmd_vel_pub_;

    // OpenCV窗口回调
    static void onMouseWrapper(int event, int x, int y, int flags, void* userdata);
    void onMouse(int event, int x, int y, int flags);

    // 回调函数
    void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthCallback(const sensor_msgs::ImageConstPtr& msg);

    // 运动控制相关
    void updateMotionControl();
    void stopRobot();
    void calculateCommand();
    void showDebugInfo(cv::Mat& display_image);

    // 跟踪器
    KCFTracker* tracker_;

    // 图像数据
    cv::Mat rgb_image_;
    cv::Mat depth_image_;

    // 跟踪状态
    bool select_flag_;
    bool begin_track_;
    bool renew_roi_;
    bool target_lost_;
    cv::Point origin_;
    cv::Rect select_rect_;
    cv::Rect track_rect_;

    // 运动控制
    double linear_speed_;
    double rotation_speed_;
    double target_distance_;  // 目标距离

    // 新增变量
    int track_lost_frames_;    // 连续跟踪失败的帧数
    int max_lost_frames_;      // 最大允许连续失败帧数
};

#endif 