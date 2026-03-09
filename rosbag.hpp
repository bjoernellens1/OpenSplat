#ifndef ROSBAG_H
#define ROSBAG_H

#include "input_data.hpp"
#include <string>

#ifdef HAVE_ROSBAG2

namespace rb {

// Load camera data from a ROS2 bag file or directory.
//
// bagPath          - Path to the ROS2 bag directory (containing metadata.yaml)
//                    or to a single .db3 / .mcap file.
// imageTopic       - sensor_msgs/Image or sensor_msgs/CompressedImage topic.
//                    Leave empty for auto-detection.
// cameraInfoTopic  - sensor_msgs/CameraInfo topic.
//                    Leave empty for auto-detection.
// poseTopic        - geometry_msgs/PoseStamped topic carrying camera poses.
//                    Leave empty to search for any pose-like topic; if none is
//                    found, identity matrices are used and a warning is printed.
InputData inputDataFromRosBag(const std::string &bagPath,
                               const std::string &imageTopic       = "",
                               const std::string &cameraInfoTopic  = "",
                               const std::string &poseTopic        = "");

} // namespace rb

#endif // HAVE_ROSBAG2

#endif // ROSBAG_H
