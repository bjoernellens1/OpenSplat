#include "opensplat_live_mapper/gaussian_scheduler.hpp"
#include <algorithm>

namespace opensplat_live {

GaussianScheduler::GaussianScheduler(const SchedulerConfig &cfg) : cfg_(cfg) {}

torch::Tensor GaussianScheduler::update(LiveGaussianMap &map,
                                         SpatialBucketIndex &index,
                                         const std::vector<BucketId> &frustumBuckets,
                                         int32_t currentFrame) {
    using namespace torch::indexing;
    int64_t N = map.size();
    if (N == 0) return torch::empty({0}, torch::kLong);

    auto &meta = map.meta();
    int64_t globalUnstable = 0;

    // Mark active buckets
    index.clearActive();
    for (BucketId bid : frustumBuckets) {
        index.markActive(bid, currentFrame);
    }

    // Update per-Gaussian flags
    auto stableAcc   = map.stableTensor_.accessor<bool,1>();
    auto unstableAcc = map.unstableTensor_.accessor<bool,1>();

    std::vector<float> priorities(static_cast<size_t>(N), 0.0f);

    for (int64_t i = 0; i < N; ++i) {
        auto &m = meta[i];

        // Age-based stability
        bool ageStable = m.age_frames >= cfg_.stableMinObservations;
        // Residual-based instability: use configurable threshold
        bool highResid = (m.mean_color_residual + m.mean_depth_residual)
                         > cfg_.residualInstabilityThresh;

        // A Gaussian is unstable if any condition is true
        bool unstable = !ageStable
                        || highResid
                        || m.confidence < cfg_.minConfidenceThresh
                        || m.frozen_until_frame > currentFrame  // frozen → skip
                        ;

        // Newly inserted or in active bucket → always unstable
        if (m.active_flag) unstable = true;

        // Cap global unstable count
        if (unstable && globalUnstable >= cfg_.maxUnstableGlobally) {
            unstable = false;
        }
        if (unstable) ++globalUnstable;

        stableAcc[i]   = !unstable;
        unstableAcc[i] =  unstable;
        m.stable_flag  = !unstable;
        m.active_flag  = false; // reset; will be set again by next append

        // Priority: newer + higher residual = higher priority
        priorities[i] = (unstable ? 1.0f : 0.0f) * (1.0f + m.mean_color_residual)
                         / static_cast<float>(std::max(1, m.age_frames));
        m.refine_priority = priorities[i];
    }

    // Collect unstable indices, sort by priority (highest first)
    std::vector<int64_t> unstableIdx;
    unstableIdx.reserve(static_cast<size_t>(globalUnstable));
    for (int64_t i = 0; i < N; ++i) {
        if (unstableAcc[i]) unstableIdx.push_back(i);
    }
    std::sort(unstableIdx.begin(), unstableIdx.end(), [&](int64_t a, int64_t b){
        return priorities[a] > priorities[b];
    });
    if (static_cast<int>(unstableIdx.size()) > cfg_.maxOptGaussians)
        unstableIdx.resize(static_cast<size_t>(cfg_.maxOptGaussians));

    auto candidateTensor = torch::from_blob(
        unstableIdx.data(),
        {static_cast<int64_t>(unstableIdx.size())},
        torch::kLong).clone();

    return candidateTensor;
}

} // namespace opensplat_live
