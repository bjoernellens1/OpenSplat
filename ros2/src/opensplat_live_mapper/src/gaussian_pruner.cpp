#include "opensplat_live_mapper/gaussian_pruner.hpp"
#include <cmath>

namespace opensplat_live {

GaussianPruner::GaussianPruner(const PrunerConfig &cfg) : cfg_(cfg) {}

int64_t GaussianPruner::prune(LiveGaussianMap &map,
                               SpatialBucketIndex &index,
                               int32_t currentFrame) {
    int64_t N = map.size();
    if (N == 0) return 0;

    auto deletedMask = torch::zeros({N}, torch::kBool);
    auto &meta = map.meta();
    auto &opacities = map.opacities(); // [N,1] logit-space

    // Compute sigmoid opacities on CPU
    auto opacitiesCpu = opacities.detach().cpu().squeeze(1); // [N]
    auto sigmoidOp = torch::sigmoid(opacitiesCpu);
    auto opAcc  = sigmoidOp.accessor<float,1>();
    auto delAcc = deletedMask.accessor<bool,1>();

    for (int64_t i = 0; i < N; ++i) {
        const auto &m = meta[i];
        bool prune = false;

        // Low opacity for long enough
        if (m.age_frames >= cfg_.graceFrames && opAcc[i] < cfg_.opacityThresh)
            prune = true;

        // Not re-observed after grace period
        if (m.age_frames >= cfg_.graceFrames
            && (currentFrame - m.last_seen_frame) > cfg_.maxUnseenFrames)
            prune = true;

        // Dynamic suspect with very low confidence
        if (m.dynamic_suspect && m.confidence < cfg_.dynamicConfirmRatio)
            prune = true;

        delAcc[i] = prune;
    }

    int64_t numPruned = deletedMask.sum().item<int64_t>();
    if (numPruned == 0) return 0;

    // Update spatial index before removing
    auto delCpu = deletedMask.to(torch::kCPU);
    auto delAccFinal = delCpu.accessor<bool,1>();
    auto meansCpu = map.means().detach().cpu(); // [N,3]
    auto meansAcc = meansCpu.accessor<float,2>();
    for (int64_t i = 0; i < N; ++i) {
        if (delAccFinal[i]) {
            std::array<float,3> xyz{meansAcc[i][0], meansAcc[i][1], meansAcc[i][2]};
            index.remove(i, xyz);
        }
    }

    map.remove(deletedMask);
    return numPruned;
}

} // namespace opensplat_live
