#include <opencv_dnn/yolov8_parser.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <opencv2/dnn.hpp>
#include <vision_msgs/create_aabb.h>

namespace opencv_dnn
{

void YOLOv8Parser::initialize()
{

}

vision_msgs::Detection2DArray YOLOv8Parser::parse(const std::vector<cv::Mat> &detections, const DetectionsContext& context)
{
  const auto& detection = detections.front();

  int class_count = detection.size[1] - 4;

  std::vector<std::vector<cv::Rect> > boxes(class_count);
  std::vector<std::vector<float> > scores(class_count);

  float x_scale = context.source_image_width/float(context.network_input_width);
  float y_scale = context.source_image_height/float(context.network_input_height);

  for(int i = 0; i < detection.size[0]; i++)
    for(int j = 0; j < detection.size[2]; j++)
    {
      auto x = detection.at<float>(i, 0, j)*x_scale;
      auto y = detection.at<float>(i, 1, j)*y_scale;
      auto width = detection.at<float>(i, 2, j)*x_scale;
      auto height = detection.at<float>(i, 3, j)*y_scale;
      cv::Rect rect(x - width/2, y - height/2, width, height);
      for(int c = 4; c < detection.size[1]; c++)
      {
        int class_id = c - 4;
        auto confidence = detection.at<float>(i,c,j);
        if(confidence > context.conf_threshold)
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

PLUGINLIB_EXPORT_CLASS(opencv_dnn::YOLOv8Parser, opencv_dnn::DetectionsParser)
