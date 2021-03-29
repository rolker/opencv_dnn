#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <darknet_ros_msgs/BoundingBoxes.h>

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

        nodeHandle.param("yolo_model/detection_classes/names", m_classLabels, std::vector<std::string>(0));
        
        nodeHandle.param("yolo_model/threshold/value", m_threshold, (float)0.3);
        
        std::string weightsModel, weightsPath;
        nodeHandle.param("yolo_model/weight_file/name", weightsModel, std::string("yolov2-tiny.weights"));
        nodeHandle.param("weights_path", weightsPath, std::string("/default"));
        weightsPath += "/" + weightsModel;
        
        std::string configModel, configPath;
        nodeHandle.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
        nodeHandle.param("config_path", configPath, std::string("/default"));
        configPath += "/" + configModel;
        
        ROS_INFO(configPath.c_str());
        ROS_INFO(weightsPath.c_str());
        m_net = cv::dnn::readNetFromDarknet(configPath, weightsPath);
        m_output_names = m_net.getUnconnectedOutLayersNames();
        
        m_boundingBoxesPublisher =
        nodeHandle.advertise<darknet_ros_msgs::BoundingBoxes>("bounding_boxes", 1, 0);
        
        ros::NodeHandle nh; // non-private node handle        

        image_transport::ImageTransport it(nh);

        m_detectionImagePublisher = it.advertise("detection_image", 1, 0);
        
        m_imageSubscriber = it.subscribe("image", 1, &ROSOpenCVDNN::imageCallback, this);
        
    }

private:
    std::vector<std::string> m_classLabels;
    float m_threshold;
    cv::dnn::Net m_net;
    std::vector<std::string> m_output_names;
    ros::Publisher m_boundingBoxesPublisher;
    image_transport::Publisher m_detectionImagePublisher;
    image_transport::Subscriber m_imageSubscriber;
    
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
        cv::dnn::blobFromImage(cv_ptr->image, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
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

        darknet_ros_msgs::BoundingBoxes boundingBoxesResults;
        boundingBoxesResults.header.stamp = ros::Time::now();
        boundingBoxesResults.header.frame_id = "detection";
        boundingBoxesResults.image_header = msg->header;

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
                
                darknet_ros_msgs::BoundingBox boundingBox;
                boundingBox.Class = m_classLabels[c];
                boundingBox.id = c;
                boundingBox.probability = scores[c][idx];
                boundingBox.xmin = rect.x;
                boundingBox.ymin = rect.y;
                boundingBox.xmax = rect.x+rect.width;
                boundingBox.ymax = rect.y+rect.height;
                boundingBoxesResults.bounding_boxes.push_back(boundingBox);
            }
        }
    
        auto total_end = std::chrono::steady_clock::now();
        
        m_boundingBoxesPublisher.publish(boundingBoxesResults);

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

        m_detectionImagePublisher.publish(cv_ptr->toImageMsg());
    }
};

std::string weightsPath;
std::string configPath;

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "darknet_ros");
    
    ROSOpenCVDNN dnn;
    
    ros::spin();
    return 0;
}    
