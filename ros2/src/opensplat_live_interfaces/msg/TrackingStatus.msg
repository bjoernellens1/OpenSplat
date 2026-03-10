# Tracking health report published by the frontend
std_msgs/Header header
uint8 STATUS_OK=0
uint8 STATUS_DEGRADED=1
uint8 STATUS_LOST=2
uint8 status
float32 confidence        # 0.0 = lost, 1.0 = perfect
float32 motion_magnitude  # metres per second
bool is_keyframe
string status_message
