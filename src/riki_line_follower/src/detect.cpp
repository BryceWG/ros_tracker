#include <cv_bridge/cv_bridge.h>
#include <cstdlib>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "ros/ros.h"
#include "ros/console.h"
#include "linedetect.hpp"
#include "riki_line_follower/pos.h"

int main(int argc, char **argv) {
    // Initializing node and object
    ros::init(argc, argv, "detection");
    ros::NodeHandle n;
    LineDetect det;
    // Creating Publisher and subscriber
    ros::Subscriber sub = n.subscribe("/kinect2/qhd/image_color",
        1, &LineDetect::imageCallback, &det); 

    ros::Publisher dirPub = n.advertise<
    riki_line_follower::pos>("direction", 1);
        riki_line_follower::pos msg;

    while (ros::ok()) {
        if (!det.img.empty()) {
            // Perform image processing
            det.img_filt = det.Gauss(det.img);
            msg.direction = det.colorthresh(det.img_filt);
            // Publish direction message
            dirPub.publish(msg);
            }
        ros::spinOnce();
    }
    // Closing image viewer
    cv::destroyWindow("Turtlebot View");
}
