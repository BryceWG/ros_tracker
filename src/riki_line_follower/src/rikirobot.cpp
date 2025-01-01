#include <geometry_msgs/Twist.h>
#include <vector>
#include "ros/ros.h"
#include "ros/console.h"
#include "rikirobot.hpp"
#include "riki_line_follower/pos.h"

void rikirobot::dir_sub(riki_line_follower::pos msg) {
    rikirobot::dir = msg.direction;
}
void rikirobot::vel_cmd(geometry_msgs::Twist &velocity,
 ros::Publisher &pub, ros::Rate &rate) {
    // If direction is left
    if (rikirobot::dir == 0) {
        velocity.linear.x = 0.05;
        velocity.angular.z = 0.1;
        pub.publish(velocity);
        rate.sleep();
        //ROS_INFO_STREAM("Turning Left");
    }
    // If direction is straight
    if (rikirobot::dir == 1) {
        velocity.linear.x = 0.35;
        velocity.angular.z = 0;
        pub.publish(velocity);
        rate.sleep();
        ROS_INFO_STREAM("Straight");
    }
    // If direction is right
    if (rikirobot::dir == 2) {
        velocity.linear.x = 0.05;
        velocity.angular.z = -0.1;
        pub.publish(velocity);
        rate.sleep();
        ROS_INFO_STREAM("Turning Right");
    }
    // If robot has to search
    if (rikirobot::dir == 3) {
        velocity.linear.x = 0;
        velocity.angular.z = -0.1;
        pub.publish(velocity);
        rate.sleep();
        velocity.linear.x = 0;
        velocity.angular.z = 0.1;
        rate.sleep();
        ROS_INFO_STREAM("Searching");
    }
}
