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
  /*detection is shape [NxQxP] where N is batch size of images input,
    Q is number of detections made per image(typically 25200), and P is the number of classes + 5*/
  int class_count = detection.size[2] - 5;
  int rows = detection.size[1];
  //create vectors to store detection rectangles and scores
  std::vector<std::vector<cv::Rect> > boxes(class_count);
  std::vector<std::vector<float> > scores(class_count);

  //scaling 
  float x_scale = context.source_image_width/float(context.network_input_width);
  float y_scale = context.source_image_height/float(context.network_input_height);

  for(int i = 0; i < detection.size[0]; i++)
  {
    for(int j = 0; j < detection.size[1]; j++)
    {
      //for each row of detections
      
      //initialize max class val and id for this row of detections
      float max_class_val = 0;
      int max_class_id; 
      auto objectness = detection.at<float>(i,j,4);

      if(objectness >= context.obj_threshold) //if objectness meets requirements
      { //parse location and make bounding box rectangle
        auto x = detection.at<float>(i, j, 0)*x_scale;
        auto y = detection.at<float>(i, j, 1)*y_scale;
        auto width = detection.at<float>(i, j, 2)*x_scale;
        auto height = detection.at<float>(i, j, 3)*y_scale;
        cv::Rect rect(x - width/2, y - height/2, width, height);

        //iterate through class probabilities
        for(int c = 5; c < detection.size[2]; c++) 
        {
          int class_id = c - 5;
          auto confidence = detection.at<float>(i,j,c);
          if ( (confidence > context.conf_threshold) && (confidence > max_class_val) )
          {
            max_class_id=class_id;
            max_class_val=confidence;
          }
        }
        boxes[max_class_id].push_back(rect);
        scores[max_class_id].push_back(max_class_val);
      }
    }
  }
 
  //non-maximum suppression to suppress multiple detections of one object
  std::vector<std::vector<int> > indices(class_count);
  for (int c = 0; c < class_count; c++)
    cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, context.nms_threshold, indices[c]);

  vision_msgs::Detection2DArray detections_msg;
  //print boxes
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
