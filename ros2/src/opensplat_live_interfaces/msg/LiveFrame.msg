# Synchronized RGB + depth frame with pose hint
std_msgs/Header header
sensor_msgs/Image rgb
sensor_msgs/Image depth
sensor_msgs/CameraInfo camera_info
geometry_msgs/PoseStamped pose_hint
float32 tracking_confidence
bool is_keyframe_candidate
