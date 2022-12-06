# ROS package for using OpenCV's DNN capability

## ROS noetic

For building on Ubuntu 20.04, an updated OpenCV (4.5) is needed.

(need from apt: libcudnn8-dev, cuda? others?)

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build
cmake ../ -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules
make -j4
sudo make install

(May need to speciy gcc-8/g++-8 for cuda)

This will install OpenCV in /usr/local

To avoid link errors, the vision_opencv package should also be built against this alternate OpenCV so clone it into your workspace.

https://github.com/rolker/vision_opencv

## Weights

Weights should be downloaded into the config folder.

From https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo

Yolo4 weights: https://drive.google.com/open?id=1L-SO373Udc9tPz5yLkgti5IAXFboVhUt