#include "visual_tracker.hpp"
#include <opencv2/imgproc/imgproc.hpp>

// KCF跟踪器参数
bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool LAB = false;

VisualTracker::VisualTracker() : it_(nh_) {
    // 初始化跟踪相关变量
    select_flag_ = false;
    begin_track_ = false;
    renew_roi_ = false;
    target_lost_ = false;
    linear_speed_ = 0.0;
    rotation_speed_ = 0.0;

    // 创建KCF跟踪器
    tracker_ = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

    // 订阅RGB和深度图像
    rgb_sub_ = it_.subscribe("/camera/rgb/image_raw", 1, &VisualTracker::rgbCallback, this);
    depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1, &VisualTracker::depthCallback, this);
    
    // 发布速度命令
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    // 创建窗口并设置鼠标回调
    cv::namedWindow("Tracking");
    cv::setMouseCallback("Tracking", onMouseWrapper, this);

    ROS_INFO("Visual tracker initialized");
}

VisualTracker::~VisualTracker() {
    delete tracker_;
    cv::destroyAllWindows();
}

void VisualTracker::rgbCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        rgb_image_ = cv_ptr->image.clone();

        // 处理选框
        if (select_flag_) {
            cv::rectangle(rgb_image_, select_rect_, cv::Scalar(255, 0, 0), 2);
        }

        // 初始化或更新跟踪器
        if (renew_roi_) {
            if (select_rect_.width > 0 && select_rect_.height > 0) {
                tracker_->init(select_rect_, rgb_image_);
                begin_track_ = true;
                target_lost_ = false;
            }
            renew_roi_ = false;
        }

        // 更新跟踪
        if (begin_track_) {
            track_rect_ = tracker_->update(rgb_image_);
            cv::rectangle(rgb_image_, track_rect_, cv::Scalar(0, 255, 255), 2);
            calculateCommand();
        }

        // 显示调试信息
        showDebugInfo(rgb_image_);
        cv::imshow("Tracking", rgb_image_);
        cv::waitKey(1);

    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void VisualTracker::depthCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        depth_image_ = cv_ptr->image.clone();
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void VisualTracker::onMouse(int event, int x, int y, int flags) {
    if (select_flag_) {
        select_rect_.x = std::min(origin_.x, x);
        select_rect_.y = std::min(origin_.y, y);
        select_rect_.width = std::abs(x - origin_.x);
        select_rect_.height = std::abs(y - origin_.y);
        select_rect_ &= cv::Rect(0, 0, rgb_image_.cols, rgb_image_.rows);
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        origin_ = cv::Point(x, y);
        select_rect_ = cv::Rect(x, y, 0, 0);
        select_flag_ = true;
        begin_track_ = false;
    } else if (event == cv::EVENT_LBUTTONUP) {
        select_flag_ = false;
        if (select_rect_.width > 0 && select_rect_.height > 0) {
            renew_roi_ = true;
        }
    }
}

void VisualTracker::onMouseWrapper(int event, int x, int y, int flags, void* userdata) {
    VisualTracker* tracker = reinterpret_cast<VisualTracker*>(userdata);
    tracker->onMouse(event, x, y, flags);
}

void VisualTracker::calculateCommand() {
    geometry_msgs::Twist cmd_vel;

    if (!begin_track_ || target_lost_) {
        cmd_vel.linear.x = 0;
        cmd_vel.angular.z = 0;
        cmd_vel_pub_.publish(cmd_vel);
        return;
    }

    // 获取目标中心点的深度值
    float depth_val = 0;
    int valid_points = 0;
    for (int i = 0; i < 5; i++) {
        int y = track_rect_.y + track_rect_.height / 3 + (i % 2) * track_rect_.height / 3;
        int x = track_rect_.x + track_rect_.width / 3 + (i / 2) * track_rect_.width / 3;
        if (y < depth_image_.rows && x < depth_image_.cols) {
            float d = depth_image_.at<float>(y, x);
            if (d > 0.1 && d < 10.0) {  // 有效深度范围0.1m到10m
                depth_val += d * 1000;  // 转换为毫米
                valid_points++;
            }
        }
    }

    if (valid_points > 0) {
        depth_val /= valid_points;
        
        // 计算线速度
        double distance_error = depth_val - MIN_DISTANCE;
        if (depth_val > MIN_DISTANCE) {
            linear_speed_ = std::min(MAX_LINEAR_SPEED, 0.3 * distance_error / 1000.0);  // 转换回米
        } else {
            linear_speed_ = std::max(-MAX_LINEAR_SPEED, 0.3 * distance_error / 1000.0);
        }

        // 计算角速度
        int center_x = track_rect_.x + track_rect_.width / 2;
        double offset = center_x - rgb_image_.cols / 2;
        rotation_speed_ = -K_ROTATION_SPEED * offset;
        rotation_speed_ = std::max(-MAX_ROTATION_SPEED, std::min(MAX_ROTATION_SPEED, rotation_speed_));

        // 检查是否丢失目标
        if (track_rect_.area() < 100 || track_rect_.area() > rgb_image_.area() / 4) {
            target_lost_ = true;
            linear_speed_ = 0;
            rotation_speed_ = 0;
        }
    } else {
        target_lost_ = true;
        linear_speed_ = 0;
        rotation_speed_ = 0;
    }

    cmd_vel.linear.x = linear_speed_;
    cmd_vel.angular.z = rotation_speed_;
    cmd_vel_pub_.publish(cmd_vel);
}

void VisualTracker::showDebugInfo(cv::Mat& display_image) {
    std::string status = target_lost_ ? "Target Lost" : (begin_track_ ? "Tracking" : "Waiting");
    cv::putText(display_image, "Status: " + status, 
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                target_lost_ ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);

    if (begin_track_ && !target_lost_) {
        // 显示速度信息
        std::stringstream ss;
        ss << "Speed: " << std::fixed << std::setprecision(2) 
           << "Linear=" << linear_speed_ << "m/s, Angular=" << rotation_speed_ << "rad/s";
        cv::putText(display_image, ss.str(), 
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 2);

        // 显示目标位置信息
        int center_x = track_rect_.x + track_rect_.width / 2;
        double offset = center_x - rgb_image_.cols / 2;
        ss.str("");
        ss << "Target: " << std::fixed << std::setprecision(1) 
           << "Offset=" << offset << "px";
        cv::putText(display_image, ss.str(), 
                    cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                    cv::Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "visual_tracker");
    VisualTracker tracker;
    ros::spin();
    return 0;
} 