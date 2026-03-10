cmake_minimum_required(VERSION 3.8)
project(opensplat_tracking_frontend)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d)
find_package(opensplat_live_interfaces REQUIRED)

add_executable(tracking_frontend_node src/tracking_frontend_node.cpp)
ament_target_dependencies(tracking_frontend_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
  opensplat_live_interfaces
)
target_link_libraries(tracking_frontend_node
  ${OpenCV_LIBS}
)

install(TARGETS tracking_frontend_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
