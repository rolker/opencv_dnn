#ifndef OPENCV_DNN_NODE_H
#define OPENCV_DNN_NODE_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <pluginlib/class_loader.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/dnn.hpp>

#include <opencv_dnn/detections_parser.h>

namespace opencv_dnn
{

class Node
{
public:
  struct TimeData
  {
    /// Time spent in network inference, in seconds.
    float inference_time;

    /// Total time including preparing the data, inference, and interpreting the results.
    float total_time;
  };


  Node();

private:

  void sendClassesTimerCallback(const ros::TimerEvent& event);
  void imageCallback(const sensor_msgs::ImageConstPtr& msg);
  void generateDetectionsImage(const sensor_msgs::ImageConstPtr& msg, cv_bridge::CvImagePtr cv_image, const vision_msgs::Detection2DArray& detections, const TimeData& time_data);

  cv::dnn::Net net_;

  int net_input_width_ = 0;
  int net_input_height_ = 0;
  std::vector<std::string> net_output_names_;


  std::string detections_parser_type_ = "opencv_dnn::YOLOv8Parser";
  boost::shared_ptr<opencv_dnn::DetectionsParser> detections_parser_;
  pluginlib::ClassLoader<opencv_dnn::DetectionsParser> parser_loader_;

  float threshold_ = 0.5;
  float nms_threshold_ = 0.4;

  ros::Publisher detections_publisher_;

  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  image_transport::Subscriber image_subscriber_;

  image_transport::Publisher detection_image_publisher_;

  std::map<int,std::string> class_labels_;
  std_msgs::String class_labels_yaml_;
  ros::Publisher class_labels_publisher_;
  ros::Timer class_labels_timer_;


};

} // namespace opencv_dnn

#endif
