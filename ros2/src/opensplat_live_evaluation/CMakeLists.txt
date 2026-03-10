cmake_minimum_required(VERSION 3.8)
project(opensplat_live_evaluation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(opensplat_live_interfaces REQUIRED)

add_executable(evaluation_node src/evaluation_node.cpp)
ament_target_dependencies(evaluation_node
  rclcpp
  nav_msgs
  geometry_msgs
  opensplat_live_interfaces
)

install(TARGETS evaluation_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
