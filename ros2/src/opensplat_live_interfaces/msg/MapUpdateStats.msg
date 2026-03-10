# Timing and quality stats for one map update cycle
std_msgs/Header header
uint32 keyframe_id
float32 visibility_pass_ms
float32 residual_pass_ms
float32 spawn_pass_ms
float32 optimize_pass_ms
float32 prune_merge_pass_ms
float32 total_update_ms
uint32 opt_gaussians_this_kf
float32 loss_rgb
float32 loss_depth
