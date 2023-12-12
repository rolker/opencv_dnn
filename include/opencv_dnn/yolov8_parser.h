#ifndef OPENCV_DNN_YOLOV8_PARSER_H
#define OPENCV_DNN_YOLOV8_PARSER_H

#include <opencv_dnn/detections_parser.h>

namespace opencv_dnn
{

class YOLOv8Parser: public DetectionsParser
{
public:
  YOLOv8Parser(){}

  void initialize() override;
  vision_msgs::Detection2DArray parse(const std::vector<cv::Mat> &detections, const DetectionsContext& context) override;

private:

};

}

#endif
