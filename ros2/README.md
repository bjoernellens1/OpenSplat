# OpenSplat Live – ROS 2 Jazzy Streaming Gaussian Mapper

This directory contains the **ROS 2 Jazzy workspace** that turns OpenSplat into
a real-time live Gaussian mapping system.  It implements the design described in
the project specification, drawing inspiration from RTG-SLAM's
stable/unstable scheduling and local active-window optimisation.

---

## Package overview

| Package | Purpose |
|---------|---------|
| `opensplat_live_interfaces` | Custom ROS 2 messages and services |
| `opensplat_live_mapper` | Core streaming Gaussian map backend |
| `opensplat_tracking_frontend` | RGB-D odometry tracking frontend |
| `opensplat_live_viewer_bridge` | RViz / Rerun visualisation bridge |
| `opensplat_live_evaluation` | Trajectory & map quality metrics |

---

## System architecture

```
RGB + Depth stream
        │
        ▼
┌─────────────────────────────┐
│  opensplat_tracking_frontend│  ← Lane A: tracking
│  (RGB-D odometry / SLAM)    │
└────────────┬────────────────┘
             │  /tracking/keyframe_candidate
             │  /tracking/odometry
             ▼
┌─────────────────────────────┐
│  opensplat_live_mapper      │  ← Lane B: streaming Gaussian map
│  LiveGaussianMap            │
│  SpatialBucketIndex         │
│  GaussianSpawner            │
│  GaussianScheduler          │
│  LocalOptimizer (online)    │
│  GaussianPruner / Merger    │
└────────────┬────────────────┘
             │  /mapper/gaussian_stats
             │  /mapper/map_update_stats
             │           │
     ┌───────┘           └──────────┐
     ▼                              ▼
┌────────────────────┐   ┌──────────────────────┐
│ viewer_bridge      │   │  Background refine   │  ← Lane C
│ (RViz / Rerun)     │   │  MapExporter (.ply)  │
└────────────────────┘   └──────────────────────┘
```

---

## Building

### Prerequisites

* ROS 2 Jazzy (or Rolling)
* LibTorch ≥ 2.0 with the same version used for OpenSplat
* OpenCV ≥ 4.5
* Eigen3

```bash
# From the repository root
cd ros2
colcon build --symlink-install \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
source install/setup.bash
```

---

## Running

### 1. Tracking frontend + live mapper

```bash
# Terminal 1 – tracking frontend
ros2 run opensplat_tracking_frontend tracking_frontend_node \
    --ros-args \
    -p kf_trans_thresh:=0.10 \
    -p kf_rot_thresh_deg:=7.0

# Terminal 2 – live mapper
ros2 run opensplat_live_mapper live_mapper_node \
    --ros-args \
    -p device:=cuda \
    -p output_dir:=/tmp/splat_output \
    -p max_spawn_per_kf:=2000 \
    -p max_opt_gaussians:=15000

# Terminal 3 – viewer bridge
ros2 run opensplat_live_viewer_bridge viewer_bridge_node

# Terminal 4 – evaluation (optional)
ros2 run opensplat_live_evaluation evaluation_node \
    --ros-args \
    -p output_dir:=/tmp/eval \
    -p gt_tum_file:=/path/to/groundtruth.txt
```

### 2. Replay a ROS 2 bag

```bash
ros2 bag play /path/to/your/bag --clock
```

### 3. Save the current map

```bash
ros2 service call /mapper/save_map \
    opensplat_live_interfaces/srv/SaveMap \
    "{output_path: '/tmp/mymap.ply'}"
```

### 4. Reset the map

```bash
ros2 service call /mapper/reset_map \
    opensplat_live_interfaces/srv/ResetMap \
    "{keep_intrinsics: true}"
```

---

## Key parameters (live mapper)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `cpu` | `cpu` or `cuda` |
| `bucket_size` | `0.30` | Spatial bucket size in metres |
| `kf_trans_thresh` | `0.10` | Keyframe translation threshold (m) |
| `kf_rot_thresh_deg` | `7.0` | Keyframe rotation threshold (°) |
| `max_spawn_per_kf` | `2000` | Gaussians spawned per keyframe |
| `max_opt_gaussians` | `15000` | Max Gaussians optimised per keyframe |
| `online_opt_steps` | `1` | Adam steps per keyframe (online) |
| `background_opt_steps` | `10` | Adam steps in background thread |
| `prune_opacity_thresh` | `0.02` | Opacity below which Gaussians are pruned |
| `online_budget_ms` | `30.0` | Wall-clock budget per keyframe (ms) |
| `snapshot_every_n_kf` | `50` | Auto-export PLY every N keyframes |

---

## Swapping the tracking frontend

The mapper backend only depends on:

* `opensplat_live_interfaces/msg/Keyframe` – pose + RGB + depth + intrinsics
* `nav_msgs/msg/Odometry` – lightweight pose stream

You can replace `opensplat_tracking_frontend` with any other frontend (ORB-SLAM3,
DROID-SLAM, GICP, …) as long as it publishes those two topics.

---

## Milestone status

| Milestone | Status |
|-----------|--------|
| M1 – playback baseline (bag → Gaussian insertion → PLY) | ✅ implemented |
| M2 – local online refinement | ✅ implemented |
| M3 – pruning / merge / freeze | ✅ implemented |
| M4 – background refinement + loop-aware updates | ✅ skeleton, loop closure TBD |

---

## License

MIT – same as the parent OpenSplat project.
