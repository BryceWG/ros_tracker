#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "geometry_msgs/Twist.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

static const std::string RGB_WINDOW = "RGB Image window";

// 优化速度控制参数
#define Max_linear_speed 0.5
#define Min_linear_speed 0.0
#define Min_distance 1.5
#define Max_distance 5.0
#define Max_rotation_speed 0.75

float linear_speed = 0;
float rotation_speed = 0;

// 优化速度计算系数
float k_linear_speed = (Max_linear_speed - Min_linear_speed) / (Max_distance - Min_distance);
float k_rotation_speed = 0.004;

// 图像中心点偏移阈值
int CENTER_OFFSET = 30;  // 中心区域宽度
int IMAGE_CENTER = 320;  // 图像中心x坐标

cv::Mat rgbimage;
cv::Rect selectRect;
cv::Point origin;
cv::Rect result;

bool select_flag = false;
bool bRenewROI = false;
bool bBeginKCF = false;

// 优化KCF参数
bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool LAB = false;

// 创建KCF跟踪器
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

void onMouse(int event, int x, int y, int, void*)
{
    if (select_flag) {
        selectRect.x = MIN(origin.x, x);        
        selectRect.y = MIN(origin.y, y);
        selectRect.width = abs(x - origin.x);   
        selectRect.height = abs(y - origin.y);
        selectRect &= cv::Rect(0, 0, rgbimage.cols, rgbimage.rows);
    }
    if (event == CV_EVENT_LBUTTONDOWN) {
        bBeginKCF = false;  
        select_flag = true; 
        origin = cv::Point(x, y);       
        selectRect = cv::Rect(x, y, 0, 0);
    }
    else if (event == CV_EVENT_LBUTTONUP) {
        select_flag = false;
        bRenewROI = true;
    }
}

class ImageConverter
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    
public:
    ros::Publisher pub;

    ImageConverter() : it_(nh_)
    {
        // 订阅RGB图像话题
        image_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, 
            &ImageConverter::imageCb, this);
        pub = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 10);

        cv::namedWindow(RGB_WINDOW);
        cv::setMouseCallback(RGB_WINDOW, onMouse, 0);
    }

    ~ImageConverter()
    {
        cv::destroyWindow(RGB_WINDOW);
    }

    void imageCb(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        rgbimage = cv_ptr->image;

        if (bRenewROI) {
            // 初始化跟踪器
            tracker.init(selectRect, rgbimage);
            bBeginKCF = true;
            bRenewROI = false;
        }

        if (bBeginKCF) {
            // 更新跟踪器
            result = tracker.update(rgbimage);
            cv::rectangle(rgbimage, result, cv::Scalar(0, 255, 0), 2);

            // 计算目标中心点
            int target_center_x = result.x + result.width/2;
            
            // 计算线速度
            float distance = result.width * result.height;  // 使用目标框大小估算距离
            if (distance > Min_distance && distance < Max_distance) {
                linear_speed = k_linear_speed * (distance - Min_distance);
            } else {
                linear_speed = 0;
            }
            
            // 限制最大线速度
            linear_speed = std::min(linear_speed, Max_linear_speed);

            // 计算角速度
            float center_offset = target_center_x - IMAGE_CENTER;
            if (abs(center_offset) < CENTER_OFFSET) {
                rotation_speed = 0;
            } else {
                rotation_speed = -k_rotation_speed * center_offset;
                rotation_speed = std::max(std::min(rotation_speed, Max_rotation_speed), -Max_rotation_speed);
            }

            ROS_INFO("Distance: %.2f, Linear: %.2f, Angular: %.2f", distance, linear_speed, rotation_speed);
        }

        // 显示跟踪框
        if (select_flag) {
            cv::rectangle(rgbimage, selectRect, cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow(RGB_WINDOW, rgbimage);
        cv::waitKey(1);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kcf_tracker");
    ImageConverter ic;
    
    ros::Rate rate(30);  // 控制循环频率
    while(ros::ok()) {
        geometry_msgs::Twist twist;
        twist.linear.x = linear_speed; 
        twist.linear.y = 0; 
        twist.linear.z = 0;
        twist.angular.x = 0; 
        twist.angular.y = 0; 
        twist.angular.z = rotation_speed;
        ic.pub.publish(twist);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}

