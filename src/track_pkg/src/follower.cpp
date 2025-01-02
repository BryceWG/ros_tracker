#include "follower.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>

Follower::Follower() {
    // 获取参数
    ros::NodeHandle private_nh("~");
    private_nh.param("max_linear_speed", max_linear_speed_, 0.5);
    private_nh.param("max_angular_speed", max_angular_speed_, 1.0);
    private_nh.param("min_distance", min_distance_, 1.0);
    private_nh.param("max_distance", max_distance_, 2.0);

    // 设置过滤器参数
    min_height_ = 0.1;  // 最小高度（m）
    max_height_ = 0.5;  // 最大高度（m）
    min_x_ = 0.5;       // 最小前向距离（m）
    max_x_ = 3.0;       // 最大前向距离（m）
    voxel_size_ = 0.02; // 体素大小（m）

    // 创建订阅者和发布者
    cloud_sub_ = nh_.subscribe("/camera/depth/points", 1, &Follower::cloudCallback, this);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    ROS_INFO("Follower node initialized");
}

void Follower::cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    // 转换为PCL点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // 过滤点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    filterCloud(cloud, filtered_cloud);

    // 如果没有点云数据，停止移动
    if (filtered_cloud->points.empty()) {
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = 0;
        cmd_vel.angular.z = 0;
        cmd_vel_pub_.publish(cmd_vel);
        return;
    }

    // 计算控制命令
    calculateCommand(filtered_cloud);
}

void Follower::filterCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                          pcl::PointCloud<pcl::PointXYZ>::Ptr& filtered_cloud) {
    // 创建临时点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 高度过滤
    pcl::PassThrough<pcl::PointXYZ> pass_z;
    pass_z.setInputCloud(cloud);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(min_height_, max_height_);
    pass_z.filter(*temp_cloud);

    // 距离过滤
    pcl::PassThrough<pcl::PointXYZ> pass_x;
    pass_x.setInputCloud(temp_cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(min_x_, max_x_);
    pass_x.filter(*filtered_cloud);

    // 聚类分割
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(filtered_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.1);  // 10cm
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(filtered_cloud);
    ec.extract(cluster_indices);

    // 找到最大的聚类（假设是前车）
    if (!cluster_indices.empty()) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr largest_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        size_t max_size = 0;
        int max_idx = 0;
        
        for (size_t i = 0; i < cluster_indices.size(); i++) {
            if (cluster_indices[i].indices.size() > max_size) {
                max_size = cluster_indices[i].indices.size();
                max_idx = i;
            }
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointIndices::Ptr indices(new pcl::PointIndices(cluster_indices[max_idx]));
        extract.setInputCloud(filtered_cloud);
        extract.setIndices(indices);
        extract.filter(*largest_cluster);
        filtered_cloud = largest_cluster;
    }
}

void Follower::calculateCommand(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    geometry_msgs::Twist cmd_vel;

    // 计算点云质心
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // 计算前向距离和横向偏移
    double forward_distance = centroid[0];  // x方向距离
    double lateral_offset = -centroid[1];   // y方向偏移（相机坐标系中，左为正）

    // 计算线速度（参考runtracker.cpp的逻辑）
    double distance_error = forward_distance - min_distance_;
    double k_linear = 0.3;  // 速度增益
    double linear_speed = k_linear * distance_error;

    // 限制线速度
    linear_speed = std::max(-max_linear_speed_, std::min(max_linear_speed_, linear_speed));

    // 计算角速度（简单的比例控制）
    double k_angular = 1.0;  // 角速度增益
    double angular_speed = k_angular * lateral_offset;
    angular_speed = std::max(-max_angular_speed_, std::min(max_angular_speed_, angular_speed));

    // 发布控制命令
    cmd_vel.linear.x = linear_speed;
    cmd_vel.angular.z = angular_speed;
    cmd_vel_pub_.publish(cmd_vel);

    ROS_INFO("Distance: %.2f m, Offset: %.2f m, Linear: %.2f m/s, Angular: %.2f rad/s",
             forward_distance, lateral_offset, linear_speed, angular_speed);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "follower_node");
    Follower follower;
    ros::spin();
    return 0;
} 