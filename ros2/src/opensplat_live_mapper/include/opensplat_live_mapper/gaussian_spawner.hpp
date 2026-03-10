#pragma once

#include <cstdint>
#include <torch/torch.h>
#include "live_gaussian_map.hpp"
#include "spatial_bucket_index.hpp"

namespace opensplat_live {

struct SpawnerConfig {
    int   maxPerKeyframe     = 2000;
    int   maxPerBucket       = 4000;
    float colorResidualThr   = 0.12f;  // pixel-normalised L1
    float depthResidualThr   = 0.05f;  // metres
    float minDepth           = 0.15f;  // metres
    float maxDepth           = 8.0f;   // metres
    int   shDegree           = 1;      // SH degree for new Gaussians
    float initOpacity        = 0.1f;
    float initScaleFromDepth = 0.005f; // tangential scale = depth * this factor
    /// Scale factor along the surface normal (fraction of tangential scale).
    /// Values < 1 produce flat, surface-hugging Gaussians.
    float normalScaleFactor  = 0.3f;
};

/// Creates new Gaussians from an RGB-D keyframe.
///
/// Pixels are classified into three groups (as described in the spec):
///   1. Newly observed  – projects into empty/weakly covered space
///   2. High colour residual
///   3. High depth residual
///
/// For each selected pixel the point is back-projected to 3-D, a local
/// normal is estimated from the depth neighbourhood, and Gaussian
/// parameters are initialised from geometry + RGB.
class GaussianSpawner {
public:
    explicit GaussianSpawner(const SpawnerConfig &cfg = {});

    struct Input {
        // [H,W,3] float32 RGB [0,1]
        torch::Tensor rgbImage;
        // [H,W]   float32 depth in metres (0 = invalid)
        torch::Tensor depthImage;
        // Camera intrinsics
        float fx, fy, cx, cy;
        // 4×4 cam-to-world (row-major, float32 CPU tensor)
        torch::Tensor camToWorld;
        // [H,W,3] float32 rendered colour residual (optional, empty = skip class 2)
        torch::Tensor colorResidual;
        // [H,W]   float32 rendered depth residual  (optional, empty = skip class 3)
        torch::Tensor depthResidual;
        // Current frame index
        int32_t frameIdx = 0;
    };

    struct Output {
        torch::Tensor means;        // [N,3] float32
        torch::Tensor scales;       // [N,3] float32 log-space
        torch::Tensor quats;        // [N,4] float32
        torch::Tensor featuresDc;   // [N,3] float32
        torch::Tensor featuresRest; // [N,(shBases-1)*3] float32
        torch::Tensor opacities;    // [N,1] float32 logit-space
        torch::Tensor normals;      // [N,3] float32
        std::vector<GaussianMeta> meta;
    };

    /// Spawn new Gaussians.  map and index are used to check occupancy.
    Output spawn(const Input &in,
                 const LiveGaussianMap &map,
                 SpatialBucketIndex &index);

private:
    SpawnerConfig cfg_;

    static torch::Tensor estimateNormals(const torch::Tensor &depth,
                                         float fx, float fy,
                                         float cx, float cy);
    static torch::Tensor backproject(const torch::Tensor &depth,
                                     const torch::Tensor &mask,
                                     float fx, float fy,
                                     float cx, float cy,
                                     const torch::Tensor &camToWorld);
};

} // namespace opensplat_live
