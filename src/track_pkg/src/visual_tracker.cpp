#include "visual_tracker.hpp"

VisualTracker::VisualTracker() : it_(nh_)
{
    // 初始化ROS订阅和发布
    rgb_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &VisualTracker::rgbCallback, this);
    depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, &VisualTracker::depthCallback, this);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    // 初始化KCF跟踪器参数
    bool HOG = true;
    bool FIXEDWINDOW = false;
    bool MULTISCALE = true;
    bool LAB = true;  // 使用LAB颜色特征
    tracker_ = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

    // 初始化状态变量
    select_flag_ = false;
    begin_track_ = false;
    renew_roi_ = false;
    target_lost_ = false;
    linear_speed_ = 0.0;
    rotation_speed_ = 0.0;
    track_lost_frames_ = 0;
    max_lost_frames_ = 10;  // 连续10帧未检测到目标则认为丢失

    // 创建窗口并设置鼠标回调
    cv::namedWindow("Tracking", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Tracking", onMouseWrapper, this);
}

VisualTracker::~VisualTracker()
{
    delete tracker_;
    cv::destroyAllWindows();
}

void VisualTracker::onMouseWrapper(int event, int x, int y, int flags, void* userdata)
{
    VisualTracker* tracker = reinterpret_cast<VisualTracker*>(userdata);
    tracker->onMouse(event, x, y, flags);
}

void VisualTracker::onMouse(int event, int x, int y, int flags)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        select_flag_ = true;
        begin_track_ = false;
        origin_ = cv::Point(x, y);
        select_rect_ = cv::Rect(x, y, 0, 0);
    }
    else if (select_flag_ && event == cv::EVENT_MOUSEMOVE)
    {
        select_rect_ = cv::Rect(origin_, cv::Point(x, y));
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        select_flag_ = false;
        if (select_rect_.width > 0 && select_rect_.height > 0)
        {
            begin_track_ = true;
            renew_roi_ = true;
        }
    }
}

void VisualTracker::rgbCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        rgb_image_ = cv_bridge::toCvShare(msg, "bgr8")->image;
        cv::Mat display_image = rgb_image_.clone();

        if (begin_track_)
        {
            if (renew_roi_)
            {
                // 确保选择框有效
                if (select_rect_.width > 20 && select_rect_.height > 20 && 
                    select_rect_.width < rgb_image_.cols/2 && 
                    select_rect_.height < rgb_image_.rows/2)
                {
                    tracker_->init(select_rect_, rgb_image_);
                    track_rect_ = select_rect_;
                    renew_roi_ = false;
                    target_lost_ = false;
                    track_lost_frames_ = 0;
                }
                else
                {
                    begin_track_ = false;
                    ROS_WARN("Invalid selection size. Please select again.");
                }
            }
            else
            {
                cv::Rect2d old_rect = track_rect_;
                track_rect_ = tracker_->update(rgb_image_);

                // 检查跟踪质量
                if (track_rect_.width <= 0 || track_rect_.height <= 0 ||
                    track_rect_.x < 0 || track_rect_.y < 0 ||
                    track_rect_.x + track_rect_.width > rgb_image_.cols ||
                    track_rect_.y + track_rect_.height > rgb_image_.rows ||
                    std::abs(track_rect_.width - old_rect.width) > old_rect.width * 0.5 ||
                    std::abs(track_rect_.height - old_rect.height) > old_rect.height * 0.5)
                {
                    track_lost_frames_++;
                    if (track_lost_frames_ > max_lost_frames_)
                    {
                        target_lost_ = true;
                        stopRobot();
                    }
                }
                else
                {
                    track_lost_frames_ = 0;
                    target_lost_ = false;
                    calculateCommand();
                }
            }

            // 显示跟踪框
            if (!target_lost_)
            {
                cv::rectangle(display_image, track_rect_, cv::Scalar(0, 255, 0), 2);
            }
        }
        else if (select_flag_)
        {
            // 显示选择框
            cv::rectangle(display_image, select_rect_, cv::Scalar(255, 0, 0), 2);
        }

        showDebugInfo(display_image);
        cv::imshow("Tracking", display_image);
        cv::waitKey(1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void VisualTracker::depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        // 将深度图像转换为CV_32FC1格式
        depth_image_ = cv_bridge::toCvShare(msg)->image;
        
        if (!begin_track_ || target_lost_ || depth_image_.empty())
        {
            return;
        }

        // 确保跟踪框在有效范围内
        int center_x = std::min(std::max(int(track_rect_.x + track_rect_.width / 2), 0), depth_image_.cols - 1);
        int center_y = std::min(std::max(int(track_rect_.y + track_rect_.height / 2), 0), depth_image_.rows - 1);
        
        // 计算采样区域的范围
        int sample_size = 5;
        int start_x = std::max(center_x - sample_size, 0);
        int end_x = std::min(center_x + sample_size, depth_image_.cols - 1);
        int start_y = std::max(center_y - sample_size, 0);
        int end_y = std::min(center_y + sample_size, depth_image_.rows - 1);

        // 计算目标区域的平均深度
        float avg_depth = 0.0f;
        int valid_points = 0;
        
        for(int y = start_y; y <= end_y; y++)
        {
            for(int x = start_x; x <= end_x; x++)
            {
                float depth = depth_image_.at<float>(y, x);
                if(!std::isnan(depth) && depth > 0.1 && depth < 10.0)  // 添加合理的深度范围检查
                {
                    avg_depth += depth;
                    valid_points++;
                }
            }
        }
        
        if(valid_points > 5)  // 确保有足够的有效点
        {
            avg_depth /= valid_points;
            target_distance_ = avg_depth;
            
            // 更新运动控制
            updateMotionControl();
        }
        else
        {
            ROS_WARN_THROTTLE(1, "Insufficient valid depth measurements (found %d points)", valid_points);
            // 不要立即将目标设为丢失，而是继续使用上一帧的深度值
            if(target_distance_ <= 0)
            {
                target_lost_ = true;
                stopRobot();
            }
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception in depth callback: %s", e.what());
    }
    catch (cv::Exception& e)
    {
        ROS_ERROR("OpenCV exception in depth callback: %s", e.what());
    }
    catch (std::exception& e)
    {
        ROS_ERROR("Standard exception in depth callback: %s", e.what());
    }
    catch (...)
    {
        ROS_ERROR("Unknown exception in depth callback");
    }
}

void VisualTracker::updateMotionControl()
{
    if (target_distance_ <= 0)
    {
        ROS_WARN_THROTTLE(1, "Invalid target distance");
        stopRobot();
        return;
    }

    // 获取参数
    double max_linear_speed, min_linear_speed;
    double min_distance, max_distance;
    double max_rotation_speed, k_rotation_speed;
    
    nh_.param("max_linear_speed", max_linear_speed, 0.5);
    nh_.param("min_linear_speed", min_linear_speed, 0.0);
    nh_.param("min_distance", min_distance, 2.5);
    nh_.param("max_distance", max_distance, 4.0);
    nh_.param("max_rotation_speed", max_rotation_speed, 0.75);
    nh_.param("k_rotation_speed", k_rotation_speed, 0.01);

    // 计算线速度
    if (target_distance_ < min_distance)
    {
        linear_speed_ = -max_linear_speed * (min_distance - target_distance_) / min_distance;
    }
    else if (target_distance_ > max_distance)
    {
        linear_speed_ = max_linear_speed * (target_distance_ - max_distance) / max_distance;
    }
    else
    {
        linear_speed_ = 0;
    }

    // 限制线速度
    linear_speed_ = std::max(-max_linear_speed, std::min(max_linear_speed, linear_speed_));

    // 计算角速度
    if (!rgb_image_.empty() && track_rect_.width > 0 && track_rect_.height > 0)
    {
        double target_center_x = track_rect_.x + track_rect_.width / 2.0;
        double image_center_x = rgb_image_.cols / 2.0;
        double error_x = target_center_x - image_center_x;
        
        rotation_speed_ = -k_rotation_speed * error_x;
        rotation_speed_ = std::max(-max_rotation_speed, std::min(max_rotation_speed, rotation_speed_));
    }
    else
    {
        rotation_speed_ = 0;
    }

    // 发布速度命令
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = linear_speed_;
    cmd_vel.angular.z = rotation_speed_;
    cmd_vel_pub_.publish(cmd_vel);
}

void VisualTracker::stopRobot()
{
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = 0;
    cmd_vel.angular.z = 0;
    cmd_vel_pub_.publish(cmd_vel);
}

void VisualTracker::calculateCommand()
{
    // 如果跟踪正常进行，运动控制会在 depthCallback 中通过 updateMotionControl 处理
    // 这里只处理异常情况
    if (target_lost_ || depth_image_.empty())
    {
        stopRobot();
    }
}

void VisualTracker::showDebugInfo(cv::Mat& display_image)
{
    // 显示跟踪状态
    std::string status;
    if (begin_track_) {
        status = target_lost_ ? "Target Lost!" : "Tracking";
    } else {
        status = "Select target with mouse";
    }
    cv::putText(display_image, status, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                target_lost_ ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);

    if (begin_track_)
    {
        // 显示运动控制信息
        std::stringstream ss;
        ss << "Speed - Linear: " << std::fixed << std::setprecision(2) << linear_speed_
           << " m/s, Angular: " << rotation_speed_ << " rad/s";
        cv::putText(display_image, ss.str(), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        // 显示目标距离
        if (!target_lost_) {
            std::stringstream ds;
            ds << "Target Distance: " << std::fixed << std::setprecision(2) << target_distance_ << " m";
            cv::putText(display_image, ds.str(), cv::Point(10, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        // 显示跟踪框中心位置
        if (!target_lost_ && track_rect_.width > 0 && track_rect_.height > 0) {
            cv::Point center(track_rect_.x + track_rect_.width/2, track_rect_.y + track_rect_.height/2);
            cv::circle(display_image, center, 3, cv::Scalar(0, 255, 255), -1);
            cv::line(display_image, 
                    cv::Point(display_image.cols/2, 0), 
                    cv::Point(display_image.cols/2, display_image.rows), 
                    cv::Scalar(255, 0, 0), 1);
        }
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "visual_tracker");
    VisualTracker tracker;
    ros::spin();
    return 0;
} 