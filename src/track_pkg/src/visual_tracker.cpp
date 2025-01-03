#include "visual_tracker.hpp"

VisualTracker::VisualTracker() : it_(nh_), tracker_(nullptr)
{
    try {
        // 初始化状态变量
        select_flag_ = false;
        begin_track_ = false;
        renew_roi_ = false;
        target_lost_ = false;
        linear_speed_ = 0.0;
        rotation_speed_ = 0.0;
        track_lost_frames_ = 0;
        max_lost_frames_ = 10;
        target_distance_ = 0.0;

        // 等待一段时间确保ROS系统完全初始化
        ros::Duration(0.5).sleep();

        // 初始化KCF跟踪器参数
        bool HOG = true;
        bool FIXEDWINDOW = false;
        bool MULTISCALE = true;
        bool LAB = true;
        tracker_ = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

        // 创建窗口并设置鼠标回调
        try {
            cv::destroyWindow("Tracking");
            cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
            cv::resizeWindow("Tracking", 640, 480);
            cv::setMouseCallback("Tracking", onMouseWrapper, this);
            cv::waitKey(100);  // 给更多时间让窗口系统创建窗口
        }
        catch (const cv::Exception& e) {
            ROS_ERROR("Failed to create OpenCV window: %s", e.what());
            throw;
        }

        // 最后初始化ROS订阅，确保其他组件都准备好
        rgb_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &VisualTracker::rgbCallback, this);
        depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, &VisualTracker::depthCallback, this);
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        
        ROS_INFO("Visual tracker initialized. Please select a target in the window.");
    }
    catch (const std::exception& e) {
        ROS_ERROR("Error in constructor: %s", e.what());
        throw;
    }
}

VisualTracker::~VisualTracker()
{
    try {
        // 停止机器人
        stopRobot();
        
        // 清理跟踪器
        if (tracker_) {
            delete tracker_;
            tracker_ = nullptr;
        }
        
        // 清理窗口
        cv::setMouseCallback("Tracking", nullptr, nullptr);
        cv::destroyAllWindows();
        
        ROS_INFO("Visual tracker cleaned up successfully");
    }
    catch (const std::exception& e) {
        ROS_ERROR("Error in destructor: %s", e.what());
    }
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
    if (!tracker_) {
        ROS_ERROR("Tracker not initialized");
        return;
    }

    try
    {
        // 使用共享指针避免不必要的拷贝
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        if (!cv_ptr || cv_ptr->image.empty()) {
            ROS_WARN_THROTTLE(1, "Empty RGB image received");
            return;
        }

        // 创建显示图像的副本
        cv::Mat display_image;
        cv_ptr->image.copyTo(display_image);

        // 确保图像大小合适
        if (display_image.cols > 1280 || display_image.rows > 720) {
            cv::resize(display_image, display_image, cv::Size(1280, 720));
        }

        // 保存原始图像用于跟踪
        rgb_image_ = cv_ptr->image;

        if (begin_track_)
        {
            if (renew_roi_)
            {
                // 确保选择框有效
                if (select_rect_.width > 20 && select_rect_.height > 20 && 
                    select_rect_.width < rgb_image_.cols/2 && 
                    select_rect_.height < rgb_image_.rows/2)
                {
                    try {
                        tracker_->init(select_rect_, rgb_image_);
                        track_rect_ = select_rect_;
                        renew_roi_ = false;
                        target_lost_ = false;
                        track_lost_frames_ = 0;
                        ROS_INFO("Target initialized successfully");
                    }
                    catch (const std::exception& e) {
                        ROS_ERROR("Failed to initialize tracker: %s", e.what());
                        begin_track_ = false;
                    }
                }
                else
                {
                    begin_track_ = false;
                    ROS_WARN("Invalid selection size. Please select again.");
                }
            }
            else
            {
                try {
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
                catch (const std::exception& e) {
                    ROS_ERROR("Error during tracking update: %s", e.what());
                    target_lost_ = true;
                    stopRobot();
                }
            }

            // 显示跟踪框
            if (!target_lost_ && track_rect_.width > 0 && track_rect_.height > 0)
            {
                cv::rectangle(display_image, track_rect_, cv::Scalar(0, 255, 0), 2);
            }
        }
        else if (select_flag_)
        {
            // 显示选择框
            cv::rectangle(display_image, select_rect_, cv::Scalar(255, 0, 0), 2);
        }

        // 添加提示信息
        cv::putText(display_image, "Press ESC to exit", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        if (!begin_track_) {
            cv::putText(display_image, "Click and drag to select target", 
                       cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cv::Scalar(0, 255, 0), 2);
        }

        showDebugInfo(display_image);
        
        try {
            if (!display_image.empty()) {
                cv::imshow("Tracking", display_image);
                char key = cv::waitKey(30);  // 增加等待时间
                if (key == 27) // ESC键
                {
                    ros::shutdown();
                }
            }
        }
        catch (const cv::Exception& e) {
            ROS_ERROR("Failed to show image: %s", e.what());
        }
    }
    catch (const cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    catch (const std::exception& e)
    {
        ROS_ERROR("Exception in rgbCallback: %s", e.what());
    }
}

void VisualTracker::depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        
        // 检查深度图像格式并进行转换
        if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
        {
            cv_ptr = cv_bridge::toCvShare(msg);
            depth_image_ = cv_ptr->image;
        }
        else if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1)
        {
            cv_ptr = cv_bridge::toCvShare(msg);
            cv::Mat temp;
            cv_ptr->image.convertTo(temp, CV_32F, 1.0/1000.0); // 转换为米为单位
            depth_image_ = temp;
        }
        else if (msg->encoding == sensor_msgs::image_encodings::RGB8 ||
                msg->encoding == sensor_msgs::image_encodings::BGR8)
        {
            // 如果收到RGB格式的深度图像，先转换为灰度图
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat gray;
            cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);
            gray.convertTo(depth_image_, CV_32F, 1.0/255.0); // 将灰度值归一化到0-1范围
        }
        else
        {
            ROS_ERROR("Unsupported depth image encoding: %s", msg->encoding.c_str());
            return;
        }

        if (!cv_ptr->image.empty())
        {
            depth_image_ = cv_ptr->image;
        }
        else
        {
            ROS_ERROR("Empty depth image received");
            return;
        }
        
        if (!begin_track_ || target_lost_)
        {
            return;
        }

        // 检查跟踪框是否有效
        if (track_rect_.width <= 0 || track_rect_.height <= 0 ||
            track_rect_.x < 0 || track_rect_.y < 0 ||
            track_rect_.x + track_rect_.width > depth_image_.cols ||
            track_rect_.y + track_rect_.height > depth_image_.rows)
        {
            ROS_WARN_THROTTLE(1, "Invalid tracking rectangle");
            return;
        }

        // 在跟踪框中心区域采样深度值
        int center_x = track_rect_.x + track_rect_.width / 2;
        int center_y = track_rect_.y + track_rect_.height / 2;
        int window_size = 2;

        std::vector<float> valid_depths;
        valid_depths.reserve((2 * window_size + 1) * (2 * window_size + 1));

        for (int dy = -window_size; dy <= window_size; dy++)
        {
            for (int dx = -window_size; dx <= window_size; dx++)
            {
                int x = center_x + dx;
                int y = center_y + dy;

                if (x >= 0 && x < depth_image_.cols && y >= 0 && y < depth_image_.rows)
                {
                    float depth = depth_image_.at<float>(y, x);
                    if (std::isfinite(depth) && depth > 0.1 && depth < 10.0)
                    {
                        valid_depths.push_back(depth);
                    }
                }
            }
        }

        if (valid_depths.size() >= 3)  // 降低有效点数要求
        {
            std::sort(valid_depths.begin(), valid_depths.end());
            target_distance_ = valid_depths[valid_depths.size() / 2];  // 使用中值
            updateMotionControl();
        }
        else
        {
            ROS_WARN_THROTTLE(1, "Insufficient valid depth measurements (found %zu points)", valid_depths.size());
            // 不要立即停止，给跟踪器一些容错时间
            track_lost_frames_++;
            if (track_lost_frames_ > max_lost_frames_)
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