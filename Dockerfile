#####################
# CUDA and ROS      #
#####################


FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04 as ros-cuda

# Install basic apt packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y locales lsb-release
RUN dpkg-reconfigure locales

# Install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-noetic-ros-base
RUN apt-get install -y --no-install-recommends python3-rosdep
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc


#####################
# CUDA, ROS, OpenCV #
#####################


FROM ros-cuda as ros-opencv

ARG OPENCV_VERSION=4.9.0
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update
RUN apt-get install -y --no-install-recommends \
        build-essential cmake \
        wget unzip 

WORKDIR /src/build

RUN wget -q --no-check-certificate https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O /src/opencv.zip
RUN wget -q --no-check-certificate https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O /src/opencv_contrib.zip

RUN unzip -qq /src/opencv.zip -d /src && rm -rf /src/opencv.zip
RUN unzip -qq /src/opencv_contrib.zip -d /src && rm -rf /src/opencv_contrib.zip

RUN cmake \
  -D OPENCV_EXTRA_MODULES_PATH=/src/opencv_contrib-${OPENCV_VERSION}/modules \
  -D OPENCV_DNN_CUDA=ON \
  -D WITH_CUDA=ON \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_opencv_python3=OFF \
  -D BUILD_TESTS=OFF \
  /src/opencv-${OPENCV_VERSION}

RUN make -j$(nproc)
RUN make install 

COPY ./entrypoint.sh /
ENTRYPOINT [ "/entrypoint.sh" ]


#####################
# opencv_dnn node   #
#####################


FROM ros-opencv as opencv_dnn

SHELL [ "/bin/bash" , "-c" ]

RUN apt-get install -y --no-install-recommends \
        git


RUN mkdir -p catkin_ws/src/opencv_dnn
WORKDIR /catkin_ws

COPY ./opencv_dnn ./src/opencv_dnn/

RUN git clone -b noetic https://github.com/ros-perception/vision_opencv.git ./src/vision_opencv
RUN git clone -b noetic-devel  https://github.com/ros-perception/image_common.git ./src/image_common

RUN source /opt/ros/noetic/setup.bash && rosdep install -i -y --from-paths src
RUN source /opt/ros/noetic/setup.bash && catkin_make

COPY ./entrypoint.sh /
ENTRYPOINT [ "/entrypoint.sh" ]


#####################
# Development Image #
#####################


FROM opencv_dnn as opencv_dnn_dev
SHELL [ "/bin/bash" , "-c" ]

ARG USERNAME=devuser
ARG UID=1000
ARG GID=${UID}

# Install extra tools for development
RUN apt-get update && apt-get install -y --no-install-recommends \
 gdb gdbserver nano less ros-noetic-usb-cam

 
# Create new user and home directory
RUN groupadd --gid $GID $USERNAME \
 && useradd --uid ${GID} --gid ${UID} --create-home ${USERNAME} \
 && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME} \
 && mkdir -p /home/${USERNAME} \
 && chown -R ${UID}:${GID} /home/${USERNAME} \
 && adduser ${USERNAME} video
 
# Set the ownership of the overlay workspace to the new user
RUN chown -R ${UID}:${GID} /catkin_ws/
 
# Set the user and source entrypoint in the user's .bashrc file
USER ${USERNAME}
RUN echo "source /entrypoint.sh" >> /home/${USERNAME}/.bashrc
RUN echo "export PS1=\"🐳 \e[0;34mopencv_dnn-dev\e[32m(\h)\e[0m \u:\w $ \"" >> /home/${USERNAME}/.bashrc