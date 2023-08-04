#include <ros/ros.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/create_aabb.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <fstream>
#include "onnx.pb.h"
#include <yaml-cpp/yaml.h>

// based on darknet_ros and
// https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49

constexpr float NMS_THRESHOLD = 0.4;

const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);

class ROSOpenCVDNN
{
public:
    ROSOpenCVDNN()
    {
        ros::NodeHandle nodeHandle("~");

        nodeHandle.param("threshold", m_threshold, (float)0.3);
        ROS_INFO_STREAM("Threshold: " << m_threshold);
        
        std::string model, model_path;
        nodeHandle.param("model", model, std::string());
        nodeHandle.param("model_path", model_path, std::string(""));
        if(!model_path.empty())
            model = model_path + "/" + model;
        
        std::string configuration, configuration_path;
        nodeHandle.param("configuration", configuration, std::string());
        nodeHandle.param("configuration_path", configuration_path, std::string(""));
        if(!configuration.empty() && !configuration_path.empty())
            configuration = configuration_path + "/" + configuration;
        

        ROS_INFO_STREAM("model: " << model);
        ROS_INFO_STREAM("configuration: " << configuration);

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
                ROS_INFO_STREAM("metadata_prop_size: " << model_proto.metadata_props_size());
                auto props = model_proto.metadata_props();
                for(auto prop: props)
                {
                    ROS_INFO_STREAM("prop: " << prop.key() << " : " << prop.value());
                    if(prop.key() == "names")
                    {
                        m_classLabelsYaml.data = prop.value();
                        auto classes = YAML::Load(prop.value());
                        if(classes.IsMap())
                        {
                            for(auto c: classes)
                            {
                                m_classLabels[c.first.as<int>()] = c.second.as<std::string>();
                            }
                        }
                    }
                }
            }

        }

        if(nodeHandle.hasParam("class_names"))
        {
            nodeHandle.param("class_names", m_classLabelsYaml.data, m_classLabelsYaml.data);
            auto classes_yaml = YAML::Load(m_classLabelsYaml.data);
            if(classes_yaml.IsMap())
            {
                m_classLabels.clear();
                for(auto c: classes_yaml)
                    m_classLabels[c.first.as<int>()] = c.second.as<std::string>();
            }
        }

        m_classLabelsPublisher = nodeHandle.advertise<std_msgs::String>("class_labels", 1, 0);


        try
        {
            if(configuration.empty())
                m_net = cv::dnn::readNet(model);
            else
                m_net = cv::dnn::readNet(model, configuration);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(1);
        }

        cv::dnn::MatShape net_input_shape;
        std::vector<cv::dnn::MatShape> in_shapes;
        std::vector<cv::dnn::MatShape> out_shapes;
        m_net.getLayerShapes(net_input_shape, 0, in_shapes, out_shapes);
        if(!in_shapes.empty())
            if(in_shapes.front().size() == 4)
            {
                m_width = in_shapes.front()[2];
                m_height = in_shapes.front()[3];
            }
        ROS_INFO_STREAM("guessed input size: " << m_width << " x " << m_height);

        nodeHandle.param("input/width", m_width, m_width);
        nodeHandle.param("input/height", m_height, m_height);
        ROS_INFO_STREAM("input size: " << m_width << " x " << m_height);

        //m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        // m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        m_output_names = m_net.getUnconnectedOutLayersNames();
        for(auto out_name: m_output_names)
            ROS_INFO_STREAM("  output name: " << out_name);

        auto unconnected = m_net.getUnconnectedOutLayers();
        for(auto l: unconnected)
        {
            ROS_INFO_STREAM("  unconnected layer: " << l);
            std::vector<cv::dnn::MatShape> in_shapes;
            std::vector<cv::dnn::MatShape> out_shapes;
            m_net.getLayerShapes(net_input_shape, l, in_shapes, out_shapes);
            for(auto out_shape: out_shapes)
            {   
                if(m_output_shape.empty())
                    m_output_shape = out_shape;
                ROS_INFO_STREAM("    " << out_shape.size() << " dimensions");
                for(auto dim: out_shape)
                    ROS_INFO_STREAM("      dim: " << dim);
            }
        }

        m_detectionsPublisher = nodeHandle.advertise<vision_msgs::Detection2DArray>("detections", 1, 0);
        
        ros::NodeHandle nh; // non-private node handle        

        m_image_transport = std::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(nh));

        m_detectionImagePublisher = m_image_transport->advertise("detection_image", 1, 0);
        
        m_imageSubscriber = m_image_transport->subscribe("image", 1, &ROSOpenCVDNN::imageCallback, this);

        m_classLabelsTimer = nodeHandle.createTimer(ros::Duration(2.0), &ROSOpenCVDNN::sendClassesTimerCallback, this);   
    }

private:
    std::map<int,std::string> m_classLabels;
    float m_threshold;
    cv::dnn::Net m_net;
    std::vector<std::string> m_output_names;
    std::vector<int> m_output_shape;
    ros::Publisher m_detectionsPublisher;
    image_transport::Publisher m_detectionImagePublisher;
    image_transport::Subscriber m_imageSubscriber;
    std::shared_ptr<image_transport::ImageTransport> m_image_transport;
    int m_width = 0;
    int m_height = 0;

    std_msgs::String m_classLabelsYaml;
    ros::Publisher m_classLabelsPublisher;
    ros::Timer m_classLabelsTimer;


    void sendClassesTimerCallback(const ros::TimerEvent& event)
    {
        m_classLabelsPublisher.publish(m_classLabelsYaml);
    }
    
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
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
        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(cv_ptr->image, blob, 0.00392, cv::Size(m_width, m_height), cv::Scalar(), false , false, CV_32F);
        m_net.setInput(blob);
        
        auto dnn_start = std::chrono::steady_clock::now();
        m_net.forward(detections, m_output_names);
        auto dnn_end = std::chrono::steady_clock::now();

        int class_count = m_output_shape[2]-5;
        std::vector<std::vector<cv::Rect> > boxes(class_count);
        std::vector<std::vector<float> > scores(class_count);
        
        float max_confidence = 0.0;

        for (auto& detection: detections)
        {
            for(int i = 0; i < m_output_shape[0]; i++)
                for(int j = 0; j < m_output_shape[1]; j++)
                {
                    max_confidence = std::max(max_confidence, detection.at<float>(i,j,4));
                    if(detection.at<float>(i,j,4) > m_threshold) // object confidence
                    {
                        //ROS_INFO_STREAM("j: " << j << " obj conf: " << detection.at<float>(i,j,4));
                        auto x = detection.at<float>(i,j, 0) * cv_ptr->image.cols/float(m_width);
                        auto y = detection.at<float>(i,j, 1) * cv_ptr->image.rows/float(m_height);
                        auto width = detection.at<float>(i,j, 2) * cv_ptr->image.cols/float(m_width);
                        auto height = detection.at<float>(i,j, 3) * cv_ptr->image.rows/float(m_height);
                        cv::Rect rect(x - width/2, y - height/2, width, height);
                        //ROS_INFO_STREAM("  x,y: " << x << "," << y << " size: " << width << "x" << height);
                        for(int class_id = 0; class_id < class_count; class_id++)
                            if(detection.at<float>(i,j,5+class_id) > m_threshold)
                            {
                               //ROS_INFO_STREAM("  class: " << class_id << " conf: " << detection.at<float>(i,j,4+class_id));
                                boxes[class_id].push_back(rect);
                                scores[class_id].push_back(detection.at<float>(i,j,5+class_id));
                            }
           
                    }
                }

        }

        //ROS_INFO_STREAM("max confidence: " << max_confidence);

        std::vector<std::vector<int> > indices(class_count);
        for (int c = 0; c < class_count; c++)
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

        vision_msgs::Detection2DArray detections_msg;
        detections_msg.header = msg->header;

        for (int c= 0; c < class_count; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];

                auto idx = indices[c][i];
                const auto& rect = boxes[c][idx];
                cv::rectangle(cv_ptr->image, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                std::ostringstream label_ss;
                if(m_classLabels.find(c) != m_classLabels.end())
                    label_ss << m_classLabels[c];
                else
                    label_ss << c;
                label_ss << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                auto label = label_ss.str();
                
                int baseline;
                auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                cv::rectangle(cv_ptr->image, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                cv::putText(cv_ptr->image, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                
                vision_msgs::Detection2D detection;
                detection.header = msg->header;
                detection.bbox = vision_msgs::createAABB2D(rect.x, rect.y, rect.width, rect.height);

                vision_msgs::ObjectHypothesisWithPose hypothesis;
                hypothesis.id = c;
                hypothesis.score = scores[c][idx];

                detection.results.push_back(hypothesis);
                detections_msg.detections.push_back(detection);
            }
        }
    
        auto total_end = std::chrono::steady_clock::now();
        
        m_detectionsPublisher.publish(detections_msg);

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();
            
        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(cv_ptr->image, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(cv_ptr->image, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

        ROS_DEBUG_STREAM(stats_ss.str());
        sensor_msgs::ImagePtr out_msg = cv_ptr->toImageMsg();
        m_detectionImagePublisher.publish(out_msg);
    }
};

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "opencv_dnn");
    
    ROSOpenCVDNN dnn;
    
    ros::spin();
    return 0;
}    
