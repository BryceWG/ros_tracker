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
#include <opencv2/imgproc/imgproc.hpp>

#include "kcftracker.hpp"

static const std::string RGB_WINDOW = "RGB Image window";
static const std::string DEPTH_WINDOW = "DEPTH Image window";

#define Max_linear_speed 0.5
#define Min_linear_speed 0
#define Min_distance 2.5
#define Max_distance 4.0
#define Max_depth_threshold 10.0  // 最大深度阈值（米）
#define Min_depth_threshold 0.4   // 最小深度阈值（米）
#define Max_rotation_speed 0.75

float linear_speed = 0;
float rotation_speed = 0;

float k_linear_speed = (Max_linear_speed - Min_linear_speed) / (Max_distance - Min_distance);
float h_linear_speed = Min_linear_speed - k_linear_speed * Min_distance;

float k_rotation_speed = 0.004;
float h_rotation_speed_left = 0.95;
float h_rotation_speed_right = 0.81;
 
int ERROR_OFFSET_X_left1 = 50;
int ERROR_OFFSET_X_left2 = 200;
int ERROR_OFFSET_X_right1 = 240;
int ERROR_OFFSET_X_right2 = 390;

cv::Mat rgbimage;
cv::Mat depthimage;
cv::Rect selectRect;
cv::Point origin;
cv::Rect result;

bool select_flag = false;
bool bRenewROI = false;  // the flag to enable the implementation of KCF algorithm for the new chosen ROI
bool bBeginKCF = false;
bool enable_get_depth = false;

bool HOG = true;
bool FIXEDWINDOW = true;
bool MULTISCALE = false;  // 关闭多尺度，提高稳定性
bool SILENT = false;
bool LAB = true;

// Create KCFTracker object
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

float dist_val[5] ;

void onMouse(int event, int x, int y, int, void*)
{
    if (select_flag)
    {
        cv::Mat temp;
        rgbimage.copyTo(temp);
        selectRect.x = MIN(origin.x, x);        
        selectRect.y = MIN(origin.y, y);
        selectRect.width = abs(x - origin.x);   
        selectRect.height = abs(y - origin.y);
        selectRect &= cv::Rect(0, 0, rgbimage.cols, rgbimage.rows);
        
        // 确保选择框不会太小
        if (selectRect.width >= 20 && selectRect.height >= 20) {
            cv::rectangle(temp, selectRect, cv::Scalar(0, 255, 0), 2, 8, 0);  // 使用绿色表示有效的选择
        } else {
            cv::rectangle(temp, selectRect, cv::Scalar(0, 0, 255), 2, 8, 0);  // 使用红色表示无效的选择
        }
        
        cv::imshow(RGB_WINDOW, temp);
        cv::waitKey(1);
    }
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        bBeginKCF = false;  
        select_flag = true; 
        origin = cv::Point(x, y);       
        selectRect = cv::Rect(x, y, 0, 0);  
    }
    else if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = false;
        if (selectRect.width >= 20 && selectRect.height >= 20)
            bRenewROI = true;
        else
            ROS_WARN("Selection too small: width=%d, height=%d (minimum is 20x20)", 
                     selectRect.width, selectRect.height);
    }
}

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber depth_sub_;
  
public:
  ros::Publisher pub;

  ImageConverter()
    : it_(nh_)
  {
    // 先创建窗口
    cv::namedWindow(RGB_WINDOW, cv::WINDOW_NORMAL);  // 改为NORMAL允许调整大小
    cv::namedWindow(DEPTH_WINDOW, cv::WINDOW_NORMAL);
    
    // 设置鼠标回调
    cv::setMouseCallback(RGB_WINDOW, onMouse, 0);

    // 订阅话题
    image_sub_ = it_.subscribe("/follower/camera/rgb/image_raw", 1, 
      &ImageConverter::imageCb, this);
    depth_sub_ = it_.subscribe("/follower/camera/depth/image_raw", 1, 
      &ImageConverter::depthCb, this);
    pub = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 1000);

    // 调整窗口大小和位置
    cv::resizeWindow(RGB_WINDOW, 800, 600);
    cv::resizeWindow(DEPTH_WINDOW, 800, 600);
    cv::moveWindow(RGB_WINDOW, 100, 100);
    cv::moveWindow(DEPTH_WINDOW, 950, 100);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(RGB_WINDOW);
    cv::destroyWindow(DEPTH_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        ROS_INFO_ONCE("RGB image received successfully");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_ptr->image.copyTo(rgbimage);

    if(rgbimage.empty()) {
        ROS_WARN("Empty RGB image received");
        return;
    }

    cv::Mat display_image;
    rgbimage.copyTo(display_image);

    if(bRenewROI)
    {
        try {
            if (selectRect.width <= 0 || selectRect.height <= 0 ||
                selectRect.x < 0 || selectRect.y < 0 ||
                selectRect.x + selectRect.width > rgbimage.cols ||
                selectRect.y + selectRect.height > rgbimage.rows)
            {
                ROS_WARN("Invalid selection rectangle or out of bounds");
                bRenewROI = false;
                return;
            }
            ROS_INFO("Initializing tracker with ROI: x=%d, y=%d, w=%d, h=%d", 
                     selectRect.x, selectRect.y, selectRect.width, selectRect.height);
            
            // 确保选择框不太小
            if (selectRect.width < 20 || selectRect.height < 20) {
                ROS_WARN("Selected region too small, minimum size is 20x20");
                bRenewROI = false;
                return;
            }
            
            tracker.init(selectRect, rgbimage);
            bBeginKCF = true;
            bRenewROI = false;
            enable_get_depth = true;
        } catch (const cv::Exception& e) {
            ROS_ERROR("Error initializing tracker: %s", e.what());
            bRenewROI = false;
            bBeginKCF = false;
            return;
        }
    }

    if(bBeginKCF)
    {
        try {
            result = tracker.update(rgbimage);
            
            // 检查跟踪框是否有效
            if (result.width <= 0 || result.height <= 0 ||
                result.x < 0 || result.y < 0 ||
                result.x + result.width > rgbimage.cols ||
                result.y + result.height > rgbimage.rows)
            {
                ROS_WARN("Tracking failed or out of bounds, reinitialize tracking");
                bBeginKCF = false;
                enable_get_depth = false;
                return;
            }
            
            cv::rectangle(display_image, result, cv::Scalar(0, 255, 255), 2, 8);
            ROS_INFO_THROTTLE(1.0, "Tracking box: x=%d, y=%d, w=%d, h=%d", 
                             result.x, result.y, result.width, result.height);
        } catch (const cv::Exception& e) {
            ROS_ERROR("Error updating tracker: %s", e.what());
            bBeginKCF = false;
            enable_get_depth = false;
            return;
        }
    }
    
    if (select_flag)
    {
        cv::rectangle(display_image, selectRect, cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    try {
        cv::imshow(RGB_WINDOW, display_image);
        cv::waitKey(1);
    } catch (const cv::Exception& e) {
        ROS_ERROR("Error displaying image: %s", e.what());
    }
  }

  void depthCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        // 尝试不同的编码格式
        if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv_ptr->image.convertTo(depthimage, CV_32FC1, 0.001);  // 转换为米为单位
            ROS_INFO_ONCE("Using 16UC1 depth encoding");
        } else {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            cv_ptr->image.copyTo(depthimage);
            ROS_INFO_ONCE("Using 32FC1 depth encoding");
        }
        
        // 对深度图像进行阈值处理
        cv::threshold(depthimage, depthimage, Max_depth_threshold, Max_depth_threshold, cv::THRESH_TRUNC);
        cv::threshold(depthimage, depthimage, Min_depth_threshold, Min_depth_threshold, cv::THRESH_TOZERO);
        
        if (!depthimage.empty()) {
            // 创建用于显示的深度图像
            cv::Mat display_depth;
            depthimage.convertTo(display_depth, CV_8UC1, 255.0/Max_depth_threshold);
            cv::applyColorMap(display_depth, display_depth, cv::COLORMAP_JET);
            cv::imshow(DEPTH_WINDOW, display_depth);
            cv::waitKey(1);
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Depth image cv_bridge exception: %s", e.what());
        return;
    }

    if(enable_get_depth && (result.width > 0 && result.height > 0))
    {
        // 添加边界检查
        if (result.x < 0 || result.y < 0 || 
            result.x + result.width > depthimage.cols || 
            result.y + result.height > depthimage.rows) {
            ROS_WARN("Tracking box out of image bounds in depth image");
            enable_get_depth = false;
            return;
        }

        // 获取深度值并进行有效性检查
        bool valid_depth = true;
        float sum_depth = 0;
        int valid_count = 0;
        
        // 在目标框内采样多个点
        const int sample_rows = 3;
        const int sample_cols = 3;
        for(int i = 0; i < sample_rows; i++) {
            for(int j = 0; j < sample_cols; j++) {
                int y = result.y + (i + 1) * result.height / (sample_rows + 1);
                int x = result.x + (j + 1) * result.width / (sample_cols + 1);
                
                float depth = depthimage.at<float>(y, x);
                if (!std::isnan(depth) && !std::isinf(depth) && 
                    depth >= Min_depth_threshold && depth <= Max_depth_threshold) {
                    sum_depth += depth;
                    valid_count++;
                    ROS_INFO_THROTTLE(1.0, "Valid depth at (%d,%d): %.3f meters", x, y, depth);
                }
            }
        }

        if (valid_count > 0) {
            float avg_depth = sum_depth / valid_count;
            ROS_INFO_THROTTLE(1.0, "Average depth from %d valid points: %.3f meters", valid_count, avg_depth);
            
            // 计算线速度
            if(avg_depth > Min_distance) {
                linear_speed = (avg_depth - Min_distance) * k_linear_speed + h_linear_speed;
            } else if(avg_depth < Min_distance) {
                linear_speed = (avg_depth - Min_distance) * k_linear_speed;
            } else {
                linear_speed = 0;
            }

            linear_speed = std::max(-Max_linear_speed, std::min(linear_speed, Max_linear_speed));
            
            // 计算角速度
            int center_x = result.x + result.width/2;
            float target_center = depthimage.cols / 2.0f;
            float angle_error = (center_x - target_center) / target_center;
            rotation_speed = -angle_error * Max_rotation_speed;
            rotation_speed = std::max(-Max_rotation_speed, std::min(rotation_speed, Max_rotation_speed));
            
            ROS_INFO_THROTTLE(1.0, "Control: linear=%.3f m/s, angular=%.3f rad/s", linear_speed, rotation_speed);
        } else {
            ROS_WARN_THROTTLE(1.0, "No valid depth measurements");
            linear_speed = 0;
            rotation_speed = 0;
        }
    }
  }
};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "kcf_tracker");
	ImageConverter ic;
  
	while(ros::ok())
	{
		ros::spinOnce();

    geometry_msgs::Twist twist;
    twist.linear.x = linear_speed; 
    twist.linear.y = 0; 
    twist.linear.z = 0;
    twist.angular.x = 0; 
    twist.angular.y = 0; 
    twist.angular.z = rotation_speed;
    ic.pub.publish(twist);

		if (cvWaitKey(33) == 'q')
      break;
	}

	return 0;
}

