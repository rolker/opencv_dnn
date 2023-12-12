cmake_minimum_required(VERSION 2.8.3)
project(opencv_dnn)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  pluginlib
  roscpp
  sensor_msgs
  vision_msgs
)

find_package(OpenCV 4.5 REQUIRED )
find_package(Protobuf REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES opencv_dnn_plugins
  CATKIN_DEPENDS  roscpp cv_bridge protobuf-dev vision_msgs image_transport
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PROTOBUF_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER src/onnx/onnx.proto)

add_executable(${PROJECT_NAME}_node src/node.cpp src/opencv_dnn.cpp ${PROTO_SRC})
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    ${Protobuf_LIBRARIES}
    yaml-cpp
)

add_library(opencv_dnn_plugins src/yolov8_parser.cpp)
add_dependencies(opencv_dnn_plugins ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(opencv_dnn_plugins ${catkin_LIBRARIES})

install(TARGETS ${PROJECT_NAME}_node
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
