#ifndef OPENCV_DNN_DETECTION_PARSER_H
#define OPENCV_DNN_DETECTION_PARSER_H

#include <vision_msgs/Detection2DArray.h>
#include <opencv2/core.hpp>

namespace opencv_dnn
{

struct DetectionsContext
{
  int source_image_width;
  int source_image_height;
  int network_input_width;
  int network_input_height;
  float threshold;
  float nms_threshold;
};

class DetectionsParser
{
  public:
    virtual ~DetectionsParser(){}
    virtual void initialize() = 0;
    virtual vision_msgs::Detection2DArray parse(const std::vector<cv::Mat> &detections, const DetectionsContext& context) = 0;
  protected:
    DetectionsParser(){}

};

} // namespace opencv_dnn

#endif
