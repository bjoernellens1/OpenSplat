cmake_minimum_required(VERSION 3.8)
project(opensplat_live_mapper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# ── ROS 2 dependencies ────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(opensplat_live_interfaces REQUIRED)

# ── Torch (LibTorch) ──────────────────────────────────────────────────────────
find_package(Torch QUIET)
if(NOT Torch_FOUND)
  message(WARNING "LibTorch not found – live_mapper will not be built")
  ament_package()
  return()
endif()

# ── Eigen (header-only) – optional ───────────────────────────────────────────
find_package(Eigen3 QUIET)

# ── Library of map/scheduling components ─────────────────────────────────────
set(MAPPER_SRCS
  src/live_gaussian_map.cpp
  src/spatial_bucket_index.cpp
  src/keyframe_manager.cpp
  src/gaussian_spawner.cpp
  src/gaussian_scheduler.cpp
  src/gaussian_pruner.cpp
  src/gaussian_merger.cpp
  src/local_optimizer.cpp
  src/map_exporter.cpp
)

add_library(opensplat_live_mapper_lib SHARED ${MAPPER_SRCS})
target_include_directories(opensplat_live_mapper_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(opensplat_live_mapper_lib PUBLIC
  ${TORCH_LIBRARIES}
)
target_compile_definitions(opensplat_live_mapper_lib PUBLIC
  ${TORCH_CXX_FLAGS}
)

# ── ROS 2 node executable ─────────────────────────────────────────────────────
add_executable(live_mapper_node src/live_mapper_node.cpp)
target_include_directories(live_mapper_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(live_mapper_node
  opensplat_live_mapper_lib
  ${TORCH_LIBRARIES}
)
ament_target_dependencies(live_mapper_node
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
  opensplat_live_interfaces
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS opensplat_live_mapper_lib live_mapper_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(opensplat_live_mapper_lib)
ament_export_dependencies(
  rclcpp sensor_msgs nav_msgs geometry_msgs cv_bridge opensplat_live_interfaces)

ament_package()
