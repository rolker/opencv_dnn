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

        nodeHandle.param("detection_classes/names", m_classLabels, std::vector<std::string>(0));
        std::stringstream label_yaml;
        label_yaml << "[";
        for(int i = 0; i < m_classLabels.size(); ++i)
        {
            label_yaml << m_classLabels[i];
            if(i < m_classLabels.size()-1)
                label_yaml << ", ";
        }
        label_yaml << "]";
        m_classLabelsYaml.data = label_yaml.str();
        m_classLabelsPublisher =
        nodeHandle.advertise<std_msgs::String>("class_labels", 1, 0);


        nodeHandle.param("threshold/value", m_threshold, (float)0.3);
        
        std::string model, model_path;
        nodeHandle.param("model", model, std::string("yolov2-tiny.weights"));
        nodeHandle.param("model_path", model_path, std::string(""));
        if(!model_path.empty())
            model = model_path + "/" + model;
        
        std::string configuration, configuration_path;
        nodeHandle.param("configuration", configuration, std::string("yolov2-tiny.cfg"));
        nodeHandle.param("configuration_path", configuration_path, std::string(""));
        if(!configuration_path.empty())
            configuration = configuration_path + "/" + configuration;
        
        nodeHandle.param("input/width", m_width, m_width);
        nodeHandle.param("input/width", m_height, m_height);

        ROS_INFO_STREAM("model: " << model);
        ROS_INFO_STREAM("configuration: " << configuration);
        ROS_INFO_STREAM("input size: " << m_width << " x " << m_height);

        try
        {
            m_net = cv::dnn::readNet(model, configuration);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(1);
        }
        
        m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        m_output_names = m_net.getUnconnectedOutLayersNames();
        
        m_detectionsPublisher =
        nodeHandle.advertise<vision_msgs::Detection2DArray>("detections", 1, 0);
        
        ros::NodeHandle nh; // non-private node handle        

        m_image_transport = std::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(nh));

        m_detectionImagePublisher = m_image_transport->advertise("detection_image", 1, 0);
        
        m_imageSubscriber = m_image_transport->subscribe("image", 1, &ROSOpenCVDNN::imageCallback, this);

        m_classLabelsTimer = nodeHandle.createTimer(ros::Duration(2.0), &ROSOpenCVDNN::sendClassesTimerCallback, this);   
    }

private:
    std::vector<std::string> m_classLabels;
    float m_threshold;
    cv::dnn::Net m_net;
    std::vector<std::string> m_output_names;
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
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat blob;
        std::vector<cv::Mat> detections;
        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(cv_ptr->image, blob, 0.00392, cv::Size(m_width, m_height), cv::Scalar(), true, false, CV_32F);
        m_net.setInput(blob);
        
        auto dnn_start = std::chrono::steady_clock::now();
        m_net.forward(detections, m_output_names);
        auto dnn_end = std::chrono::steady_clock::now();
        
        std::vector<std::vector<int> > indices;
        indices.resize(m_classLabels.size());
        std::vector<std::vector<cv::Rect> > boxes;
        boxes.resize(m_classLabels.size());
        std::vector<std::vector<float> > scores;
        scores.resize(m_classLabels.size());
        
        for (auto& output : detections)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                auto x = output.at<float>(i, 0) * cv_ptr->image.cols;
                auto y = output.at<float>(i, 1) * cv_ptr->image.rows;
                auto width = output.at<float>(i, 2) * cv_ptr->image.cols;
                auto height = output.at<float>(i, 3) * cv_ptr->image.rows;
                cv::Rect rect(x - width/2, y - height/2, width, height);

                for (int c = 0; c < m_classLabels.size(); c++)
                {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= m_threshold)
                    {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < m_classLabels.size(); c++)
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

        vision_msgs::Detection2DArray detections_msg;
        detections_msg.header = msg->header;

        for (int c= 0; c < m_classLabels.size(); c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];

                auto idx = indices[c][i];
                const auto& rect = boxes[c][idx];
                cv::rectangle(cv_ptr->image, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                std::ostringstream label_ss;
                label_ss << m_classLabels[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
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

std::string weightsPath;
std::string configPath;

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "opencv_dnn");
    
    ROSOpenCVDNN dnn;
    
    ros::spin();
    return 0;
}    
