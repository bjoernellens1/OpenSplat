#pragma once

#include <cstdint>
#include <chrono>
#include <torch/torch.h>
#include "live_gaussian_map.hpp"
#include "spatial_bucket_index.hpp"

namespace opensplat_live {

struct SchedulerConfig {
    /// Observation count required before a Gaussian is considered stable.
    int   stableMinObservations  = 8;
    /// Residual quantile above which a Gaussian is kept unstable.
    float unstableResidualQuantile = 0.80f;
    /// Combined residual threshold (color+depth) above which a Gaussian is unstable.
    float residualInstabilityThresh = 0.05f;
    /// Minimum confidence below which a Gaussian is always unstable.
    float minConfidenceThresh    = 0.3f;
    /// Maximum number of Gaussians submitted to the local optimizer per KF.
    int   maxOptGaussians        = 15000;
    /// Maximum total unstable Gaussians at any time.
    int   maxUnstableGlobally    = 50000;
    /// Hard cap on Gaussians per bucket.
    int   maxPerBucket           = 4000;
};

/// Computes active-bucket sets, stable/unstable masks and refine priorities.
///
/// Must be called once per keyframe before LocalOptimizer.
class GaussianScheduler {
public:
    explicit GaussianScheduler(const SchedulerConfig &cfg = {});

    /// Update stable/unstable flags and active-bucket set.
    /// Returns indices selected for optimization this keyframe.
    torch::Tensor update(LiveGaussianMap &map,
                         SpatialBucketIndex &index,
                         const std::vector<BucketId> &frustumBuckets,
                         int32_t currentFrame);

    const SchedulerConfig& config() const { return cfg_; }

private:
    SchedulerConfig cfg_;
};

} // namespace opensplat_live
