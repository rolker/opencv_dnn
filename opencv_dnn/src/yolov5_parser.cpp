#include <opencv_dnn/yolov5_parser.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <opencv2/dnn.hpp>
#include <vision_msgs/create_aabb.h>

namespace opencv_dnn
{

void YOLOv5Parser::initialize()
{

}

vision_msgs::Detection2DArray YOLOv5Parser::parse(const std::vector<cv::Mat> &detections, const DetectionsContext& context)
{
  const auto& detection = detections.front();

  int class_count = detection.size[2] - 5;

  std::vector<std::vector<cv::Rect> > boxes(class_count);
  std::vector<std::vector<float> > scores(class_count);

  float x_scale = context.source_image_width/float(context.network_input_width);
  float y_scale = context.source_image_height/float(context.network_input_height);

  for(int i = 0; i < detection.size[0]; i++)
    for(int j = 0; j < detection.size[1]; j++)
    {
      auto x = detection.at<float>(i, j, 0)*x_scale;
      auto y = detection.at<float>(i, j, 1)*y_scale;
      auto width = detection.at<float>(i, j, 2)*x_scale;
      auto height = detection.at<float>(i, j, 3)*y_scale;
      cv::Rect rect(x - width/2, y - height/2, width, height);
      auto objectness = detection.at<float>(i, j, 4);
      if(objectness >= context.threshold)
        for(int c = 5; c < detection.size[2]; c++)
        {
          int class_id = c - 5;
          auto confidence = detection.at<float>(i,j,c);
          if(confidence > context.threshold)
          {
            boxes[class_id].push_back(rect);
            scores[class_id].push_back(confidence);
          }
        }
    }

  std::vector<std::vector<int> > indices(class_count);
  for (int c = 0; c < class_count; c++)
    cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, context.nms_threshold, indices[c]);

  vision_msgs::Detection2DArray detections_msg;

  for (int c= 0; c < class_count; c++)
  {
    for (size_t i = 0; i < indices[c].size(); ++i)
    {
      auto idx = indices[c][i];
      const auto& rect = boxes[c][idx];
      
      vision_msgs::Detection2D detection;
      detection.bbox = vision_msgs::createAABB2D(rect.x, rect.y, rect.width, rect.height);

      vision_msgs::ObjectHypothesisWithPose hypothesis;
      hypothesis.id = c;
      hypothesis.score = scores[c][idx];

      detection.results.push_back(hypothesis);
      detections_msg.detections.push_back(detection);
    }
  }

  return detections_msg;

}


} // namespace opencv_dnn

PLUGINLIB_EXPORT_CLASS(opencv_dnn::YOLOv5Parser, opencv_dnn::DetectionsParser)
