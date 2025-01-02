#include "visual_tracker.hpp"

VisualTracker::VisualTracker() : it_(nh_)
{
    // 初始化ROS订阅和发布
    rgb_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &VisualTracker::rgbCallback, this);
    depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, &VisualTracker::depthCallback, this);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    // 初始化跟踪器
    tracker_ = new KCFTracker(true, true, true, false);

    // 初始化状态变量
    select_flag_ = false;
    begin_track_ = false;
    renew_roi_ = false;
    target_lost_ = false;
    linear_speed_ = 0.0;
    rotation_speed_ = 0.0;

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
                tracker_->init(select_rect_, rgb_image_);
                track_rect_ = select_rect_;
                renew_roi_ = false;
                target_lost_ = false;
            }
            else
            {
                track_rect_ = tracker_->update(rgb_image_);
                calculateCommand();
            }

            // 显示跟踪框
            cv::rectangle(display_image, track_rect_, cv::Scalar(0, 255, 0), 2);
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
        cv::Mat depth_image = cv_bridge::toCvShare(msg)->image;
        
        if (begin_track_ && !target_lost_)
        {
            // 获取跟踪框中心点的深度值
            int center_x = track_rect_.x + track_rect_.width / 2;
            int center_y = track_rect_.y + track_rect_.height / 2;
            
            // 计算目标区域的平均深度
            float avg_depth = 0.0f;
            int valid_points = 0;
            
            // 在目标框中心区域采样深度值
            int sample_size = 5;
            for(int i = -sample_size; i <= sample_size; i++)
            {
                for(int j = -sample_size; j <= sample_size; j++)
                {
                    int x = center_x + i;
                    int y = center_y + j;
                    
                    if(x >= 0 && x < depth_image.cols && y >= 0 && y < depth_image.rows)
                    {
                        float depth = depth_image.at<float>(y, x);
                        if(!std::isnan(depth) && depth > 0)
                        {
                            avg_depth += depth;
                            valid_points++;
                        }
                    }
                }
            }
            
            if(valid_points > 0)
            {
                avg_depth /= valid_points;
                target_distance_ = avg_depth;
                
                // 更新运动控制
                updateMotionControl();
            }
            else
            {
                ROS_WARN("No valid depth measurements for target");
                target_lost_ = true;
                stopRobot();
            }
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception in depth callback: %s", e.what());
    }
}

void VisualTracker::updateMotionControl()
{
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

    // 计算角速度
    double target_center_x = track_rect_.x + track_rect_.width / 2.0;
    double image_center_x = rgb_image_.cols / 2.0;
    double error_x = target_center_x - image_center_x;
    
    rotation_speed_ = -k_rotation_speed * error_x;
    rotation_speed_ = std::max(-max_rotation_speed, std::min(max_rotation_speed, rotation_speed_));

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