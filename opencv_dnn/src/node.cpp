#include <opencv_dnn/node.h>

#include <fstream>
#include "onnx.pb.h"
#include <yaml-cpp/yaml.h>

namespace opencv_dnn
{

Node::Node()
  :parser_loader_("opencv_dnn", "opencv_dnn::DetectionsParser")
{
  ros::NodeHandle pnh("~");
  pnh.param("detections_parser", detections_parser_type_, detections_parser_type_);

  try
  {
    detections_parser_ = parser_loader_.createInstance(detections_parser_type_);
    detections_parser_->initialize();
  }
  catch(const std::exception& e)
  {
    ROS_ERROR_STREAM("Failed to load detections parser: " << detections_parser_type_ << " error: " << e.what());
  }

  pnh.param("threshold", threshold_, threshold_);
  pnh.param("nms_threshold", nms_threshold_, nms_threshold_);

  std::string model, model_path;
  pnh.param("model", model, std::string());
  pnh.param("model_path", model_path, std::string(""));
  if(!model_path.empty())
    model = model_path + "/" + model;
        
  std::string configuration, configuration_path;
  pnh.param("configuration", configuration, std::string());
  pnh.param("configuration_path", configuration_path, std::string(""));
  if(!configuration.empty() && !configuration_path.empty())
    configuration = configuration_path + "/" + configuration;

  ROS_INFO_STREAM("model: " << model);
  ROS_INFO_STREAM("config: " << configuration);
  ROS_INFO_STREAM("threshold: " << threshold_ << " nms threshold: " << nms_threshold_ << " parser: " << detections_parser_type_);

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if(model.rfind(".onnx") == model.size()-5)
  {
    ROS_INFO_STREAM("Looking for class labels in .onnx file");
    std::fstream input(model, std::ios::in | std::ios::binary);
    if(!input)
        ROS_WARN_STREAM("Unable to open " << model);
    else
    {
      input.seekg(0, std::ios::end);
      size_t length = input.tellg();
      input.seekg(0, std::ios::beg);
      std::vector<char> data(length);
      input.read(data.data(), data.size());
      onnx::ModelProto model_proto;
      if(!model_proto.ParseFromArray(data.data(), data.size()))
          ROS_WARN_STREAM("Unable to parse " << model);
      //ROS_INFO_STREAM("metadata_prop_size: " << model_proto.metadata_props_size());
      auto props = model_proto.metadata_props();
      for(auto prop: props)
      {
        //ROS_INFO_STREAM("prop: " << prop.key() << " : " << prop.value());
        if(prop.key() == "names")
        {
          class_labels_yaml_.data = prop.value();
          auto classes = YAML::Load(prop.value());
          if(classes.IsMap())
          {
            for(auto c: classes)
            {
              class_labels_[c.first.as<int>()] = c.second.as<std::string>();
            }
          }
        }
      }
    }

  }

  if(pnh.hasParam("class_names"))
  {
    pnh.param("class_names", class_labels_yaml_.data, class_labels_yaml_.data);
    auto classes_yaml = YAML::Load(class_labels_yaml_.data);
    if(classes_yaml.IsMap())
    {
      class_labels_.clear();
      for(auto c: classes_yaml)
        class_labels_[c.first.as<int>()] = c.second.as<std::string>();
    }
  }

  class_labels_publisher_ = pnh.advertise<std_msgs::String>("class_labels", 1, 0);

  try
  {
    if(configuration.empty())
      net_ = cv::dnn::readNet(model);
    else
      net_ = cv::dnn::readNet(model, configuration);
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    exit(1);
  }

  bool using_cuda = false;
  auto backends = cv::dnn::getAvailableBackends();
  for (auto& backend: backends)
  {
    ROS_INFO_STREAM("backend/target" << backend.first << " " << backend.second);
    if( backend.first == cv::dnn::DNN_BACKEND_CUDA && backend.second == cv::dnn::DNN_TARGET_CUDA)
    {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
      using_cuda = true;
    }
  }


  cv::dnn::MatShape net_input_shape;
  std::vector<cv::dnn::MatShape> in_shapes;
  std::vector<cv::dnn::MatShape> out_shapes;
  net_.getLayerShapes(net_input_shape, 0, in_shapes, out_shapes);
  if(!in_shapes.empty())
    if(in_shapes.front().size() == 4)
    {
        net_input_width_= in_shapes.front()[2];
        net_input_height_ = in_shapes.front()[3];
    }
  ROS_INFO_STREAM("guessed input size: " << net_input_width_ << " x " << net_input_height_);

  pnh.param("input/width", net_input_width_, net_input_width_);
  pnh.param("input/height", net_input_height_, net_input_height_);
  ROS_INFO_STREAM("input size: " << net_input_width_ << " x " << net_input_height_);

  net_output_names_ = net_.getUnconnectedOutLayersNames();
  for(auto out_name: net_output_names_)
    ROS_INFO_STREAM("  net output name: " << out_name);


  if(using_cuda)
  {
    ROS_INFO_STREAM("CUDA detected so testing...");
    cv::Mat blank_image(net_input_width_, net_input_height_, CV_8UC3, cv::Scalar(0,0,255));
    cv::Mat blob;
    std::vector<cv::Mat> detections;
    cv::dnn::blobFromImage(blank_image, blob, 0.00392, cv::Size(net_input_width_, net_input_height_), cv::Scalar(), false , false, CV_32F);
    net_.setInput(blob);
    try
    {
      net_.forward(detections, net_output_names_);
      ROS_INFO_STREAM("CUDA seems to work!");
    }
    catch (cv::Exception)
    {
      ROS_INFO_STREAM("CUDA does NOT seem to work.");
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    }
  }


  detections_publisher_ = pnh.advertise<vision_msgs::Detection2DArray>("detections", 1, 0);
        
  ros::NodeHandle nh; // non-private node handle        

  image_transport_ = std::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(nh));

  image_subscriber_ = image_transport_->subscribe("image", 1, &Node::imageCallback, this);

  class_labels_timer_ = pnh.createTimer(ros::Duration(2.0), &Node::sendClassesTimerCallback, this);   

  detection_image_publisher_ = image_transport_->advertise("detection_image", 1);
}

void Node::sendClassesTimerCallback(const ros::TimerEvent& event)
{
  class_labels_publisher_.publish(class_labels_yaml_);
}
    
void Node::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  auto total_start = std::chrono::steady_clock::now();
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
  }
  catch (cv_bridge::Exception& e)
  {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
  }

  cv::Mat blob;
  std::vector<cv::Mat> detections;
  cv::dnn::blobFromImage(cv_ptr->image, blob, 0.00392, cv::Size(net_input_width_, net_input_height_), cv::Scalar(), false , false, CV_32F);
  net_.setInput(blob);
    
  auto dnn_start = std::chrono::steady_clock::now();
  net_.forward(detections, net_output_names_);
  auto dnn_end = std::chrono::steady_clock::now();

  opencv_dnn::DetectionsContext context;
  context.threshold = threshold_;
  context.nms_threshold = nms_threshold_;
  context.source_image_width = msg->width;
  context.source_image_height = msg->height;
  context.network_input_width = net_input_width_;
  context.network_input_height = net_input_height_;

  vision_msgs::Detection2DArray detections_msg = detections_parser_->parse(detections, context);

  auto total_end = std::chrono::steady_clock::now();
  detections_publisher_.publish(detections_msg);
  TimeData td;
  td.inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count()/1000.0;
  td.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count()/1000.0;

  generateDetectionsImage(msg, cv_ptr, detections_msg, td);
  
}
  

void Node::generateDetectionsImage(const sensor_msgs::ImageConstPtr& msg, cv_bridge::CvImagePtr cv_image, const vision_msgs::Detection2DArray& detections, const TimeData& time_data)
{
  const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
  };
  const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);

  for(const auto& detection: detections.detections)
  {
    if(!detection.results.empty())
    {
      auto detection_class = detection.results.front().id;
      const auto color = colors[detection_class % NUM_COLORS];
      auto center = detection.bbox.center;
      auto half_width = detection.bbox.size_x/2.0;
      auto half_height = detection.bbox.size_y/2.0;
      cv::rectangle(cv_image->image, cv::Point(center.x-half_width, center.y-half_height), cv::Point(center.x+half_width, center.y+half_height), color, 3);

      std::ostringstream label_ss;
      if(class_labels_.find(detection_class) != class_labels_.end())
        label_ss << class_labels_[detection_class];
      else
        label_ss << detection_class;
      label_ss << ": " << std::fixed << std::setprecision(2) << detection.results.front().score;
      auto label = label_ss.str();
      
      int baseline;
      auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
      cv::rectangle(cv_image->image, cv::Point(center.x-half_width, center.y-half_height - label_bg_sz.height - baseline - 10), cv::Point(center.x+half_width + label_bg_sz.width, center.y-half_height), color, cv::FILLED);
      cv::putText(cv_image->image, label.c_str(), cv::Point(center.x-half_width, center.y-half_height - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));

    }
  }

  std::ostringstream stats_ss;
  stats_ss << std::fixed << std::setprecision(2);
  stats_ss << "Inference FPS: " << 1.0/time_data.inference_time << ", Total FPS: " << 1.0/time_data.total_time;
  auto stats = stats_ss.str();
      
  int baseline;
  auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
  cv::rectangle(cv_image->image, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(cv_image->image, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

  ROS_DEBUG_STREAM(stats_ss.str());
  sensor_msgs::ImagePtr out_msg = cv_image->toImageMsg();
  detection_image_publisher_.publish(out_msg);
}



} // namespace opencv_dnn

