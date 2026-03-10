# A selected keyframe carrying pose, image, depth and sparse support
std_msgs/Header header
uint32 keyframe_id
geometry_msgs/PoseStamped pose
sensor_msgs/Image rgb
sensor_msgs/Image depth
sensor_msgs/CameraInfo camera_info
float32 motion_magnitude
float32 scene_overlap
float32[36] pose_covariance        # 6×6 row-major
