#pragma once

#include <cstdint>
#include <torch/torch.h>
#include "live_gaussian_map.hpp"
#include "spatial_bucket_index.hpp"

namespace opensplat_live {

struct PrunerConfig {
    float opacityThresh       = 0.02f;  // prune if sigmoid(opacity) < this
    int   graceFrames         = 10;     // don't prune before this many frames
    int   maxUnseenFrames     = 30;     // prune if not seen for this many frames
    float dynamicConfirmRatio = 0.1f;   // dynamic suspect → prune if below
};

/// Removes redundant or invalid Gaussians from the map.
class GaussianPruner {
public:
    explicit GaussianPruner(const PrunerConfig &cfg = {});

    /// Returns the number of Gaussians removed.
    int64_t prune(LiveGaussianMap &map,
                  SpatialBucketIndex &index,
                  int32_t currentFrame);

private:
    PrunerConfig cfg_;
};

} // namespace opensplat_live
