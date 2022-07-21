# ROS package for using OpenCV's DNN capability

## ROS noetic

For building on Ubuntu 20.04, an updated OpenCV (4.5) is needed.

git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build
cd build
cmake ../
make -j4
sudo make install

This will install OpenCV in /usr/local

To avoid link errors, the vision_opencv package should also be built against this alternate OpenCV so clone it into your workspace.

https://github.com/rolker/vision_opencv

## Weights

Weights should be downloaded into the config folder.

