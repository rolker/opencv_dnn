#!/bin/bash

source /opt/ros/noetic/setup.bash

if [ -f /catkin_ws/devel/setup.bash ]
then
  echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
  source /catkin_ws/devel/setup.bash
fi


exec "$@"
