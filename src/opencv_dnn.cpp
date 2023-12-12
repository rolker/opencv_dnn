#include <ros/ros.h>
#include <opencv_dnn/node.h>

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "opencv_dnn");
    
    opencv_dnn::Node dnn;
    
    ros::spin();
    return 0;
}    
