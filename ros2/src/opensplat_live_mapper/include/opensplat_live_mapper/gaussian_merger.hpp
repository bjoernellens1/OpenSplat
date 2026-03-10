#pragma once

#include <cstdint>
#include <torch/torch.h>
#include "live_gaussian_map.hpp"
#include "spatial_bucket_index.hpp"

namespace opensplat_live {

struct MergerConfig {
    float distanceThreshFactor = 1.0f;   // merge if dist < factor * local scale
    float normalAngleThreshDeg = 20.0f;
    float colorDistThresh      = 0.08f;  // L2 distance in [0,1] RGB
    int   maxMergesPerKf       = 500;
};

/// Merges near-identical neighbouring Gaussians to keep count bounded.
class GaussianMerger {
public:
    explicit GaussianMerger(const MergerConfig &cfg = {});

    /// Returns the number of Gaussians merged (removed).
    int64_t merge(LiveGaussianMap &map,
                  SpatialBucketIndex &index,
                  int32_t currentFrame);

private:
    MergerConfig cfg_;
};

} // namespace opensplat_live
