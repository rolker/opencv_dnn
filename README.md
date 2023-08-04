# ROS package for using OpenCV's DNN capability

## ROS noetic

For building on Ubuntu 20.04, an updated OpenCV (4.5+) is needed.

### Cuda

If using cuda, you may want to install the latest version.

To see what's installed:

    apt list --installed | grep cuda

From https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu

    sudo apt-key del 7fa2af80
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update

As of this writing, OpenCV will not build using Cuda 12.2, so install 12.1.

    sudo apt install cuda-12-1

Remove metapackage that installs latest cuda:

    sudo apt remove cuda
    sudo apt autoremove

Add following to ~/.bashrc and make sure to source it or reboot.

    export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}

### Clone OpenCV from github

    cd ~/src
    git clone https://github.com/opencv/opencv.git
    git clone https://github.com/opencv/opencv_contrib.git
    cd opencv
    mkdir build
    cd build

Without cuda:

    cmake ../ -D CMAKE_INSTALL_PREFIX=~/.local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules

With cuda:

    cmake ../ -D CMAKE_INSTALL_PREFIX=~/.local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D OPENCV_DNN_CUDA=ON -D WITH_CUDA=ON


    make -j4
    make install

This will install OpenCV in ~/.local

To avoid link errors, the vision_opencv package should also be built against this alternate OpenCV so clone it into your workspace.

https://github.com/rolker/vision_opencv

## Weights

See onnx model zoo

## Parsing class labels from ONNX 

The src/onnx/onnx.proto file was copied from: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
