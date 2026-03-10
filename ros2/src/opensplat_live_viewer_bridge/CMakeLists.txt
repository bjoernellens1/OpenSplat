cmake_minimum_required(VERSION 3.8)
project(opensplat_live_viewer_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(opensplat_live_interfaces REQUIRED)

add_executable(viewer_bridge_node src/viewer_bridge_node.cpp)
ament_target_dependencies(viewer_bridge_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  opensplat_live_interfaces
)

install(TARGETS viewer_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
