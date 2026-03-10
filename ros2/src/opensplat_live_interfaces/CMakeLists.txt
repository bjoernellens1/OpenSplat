cmake_minimum_required(VERSION 3.8)
project(opensplat_live_interfaces)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LiveFrame.msg"
  "msg/Keyframe.msg"
  "msg/GaussianStats.msg"
  "msg/MapUpdateStats.msg"
  "msg/TrackingStatus.msg"
  "srv/SaveMap.srv"
  "srv/ResetMap.srv"
  "srv/SetMapperMode.srv"
  "srv/TriggerRefinement.srv"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
