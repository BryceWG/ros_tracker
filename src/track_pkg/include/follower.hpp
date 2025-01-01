#ifndef FOLLOWER_HPP
#define FOLLOWER_HPP

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Twist.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/centroid.h>

class Follower {
public:
    Follower();
    ~Follower() = default;

private:
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg);
    void filterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud);
    void calculateCommand(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher cmd_vel_pub_;

    // 参数
    double max_linear_speed_;
    double max_angular_speed_;
    double min_distance_;
    double max_distance_;

    // 过滤器参数
    double min_height_;
    double max_height_;
    double min_x_;
    double max_x_;
    double voxel_size_;
};

#endif // FOLLOWER_HPP 