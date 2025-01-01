#include <cv_bridge/cv_bridge.h>
#include <cstdlib>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "ros/ros.h"
#include "ros/console.h"
#include "rikirobot.hpp"
#include "riki_line_follower/pos.h"

int main(int argc, char **argv) {
    // Initializing node and object
    ros::init(argc, argv, "Velocity");
    ros::NodeHandle n;
    rikirobot bot;
    geometry_msgs::Twist velocity;
    // Creating subscriber and publisher
    ros::Subscriber sub = n.subscribe("/direction",
        1, &rikirobot::dir_sub, &bot);
    ros::Publisher pub = n.advertise<geometry_msgs::Twist>
        ("/cmd_vel", 10);
    ros::Rate rate(20);
    while (ros::ok()) {
        ros::spinOnce();
        // Publish velocity commands to turtlebot
        bot.vel_cmd(velocity, pub, rate);
        rate.sleep();
    }
    return 0;
}
