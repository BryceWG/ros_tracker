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
        depth_image_ = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void VisualTracker::calculateCommand()
{
    if (!depth_image_.empty() && !target_lost_)
    {
        // 获取目标中心点的深度值
        cv::Point center(track_rect_.x + track_rect_.width/2, track_rect_.y + track_rect_.height/2);
        float depth = depth_image_.at<float>(center);

        // 计算线速度
        if (depth > MAX_DISTANCE)
        {
            linear_speed_ = MAX_LINEAR_SPEED;
        }
        else if (depth < MIN_DISTANCE)
        {
            linear_speed_ = -MAX_LINEAR_SPEED;
        }
        else
        {
            linear_speed_ = 0.0;
        }

        // 计算角速度
        double image_center = rgb_image_.cols / 2.0;
        double target_center = center.x;
        double error = target_center - image_center;
        rotation_speed_ = -K_ROTATION_SPEED * error;

        // 限制角速度
        rotation_speed_ = std::max(-MAX_ROTATION_SPEED, std::min(rotation_speed_, MAX_ROTATION_SPEED));

        // 发布速度命令
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = linear_speed_;
        cmd_vel.angular.z = rotation_speed_;
        cmd_vel_pub_.publish(cmd_vel);
    }
    else
    {
        // 目标丢失，停止运动
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = 0.0;
        cmd_vel_pub_.publish(cmd_vel);
    }
}

void VisualTracker::showDebugInfo(cv::Mat& display_image)
{
    std::string status = begin_track_ ? "Tracking" : "Select target";
    cv::putText(display_image, status, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    if (begin_track_)
    {
        std::stringstream ss;
        ss << "Speed - Linear: " << std::fixed << std::setprecision(2) << linear_speed_
           << " Angular: " << rotation_speed_;
        cv::putText(display_image, ss.str(), cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "visual_tracker");
    VisualTracker tracker;
    ros::spin();
    return 0;
} 