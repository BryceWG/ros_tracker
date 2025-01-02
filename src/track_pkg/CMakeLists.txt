cmake_minimum_required(VERSION 3.0.2)
project(track_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  gazebo_ros
  pcl_ros
  pcl_conversions
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS 
    roscpp 
    rospy 
    std_msgs 
    sensor_msgs 
    geometry_msgs 
    gazebo_ros 
    pcl_ros
    cv_bridge
    image_transport
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 添加follower_node可执行文件
add_executable(follower_node src/follower.cpp)
target_link_libraries(follower_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# 添加visual_tracker节点
add_executable(visual_tracker_node 
  src/visual_tracker.cpp
)

target_link_libraries(visual_tracker_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

# Install URDF files
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
  FILES_MATCHING PATTERN "*.urdf.xacro"
)

# Install world files
install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
  FILES_MATCHING PATTERN "*.world"
)


