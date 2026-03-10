#pragma once

#include <cstdint>
#include <chrono>
#include <torch/torch.h>
#include "live_gaussian_map.hpp"

namespace opensplat_live {

struct OptimizerConfig {
    int   onlineSteps    = 1;
    int   backgroundSteps= 10;
    float lrMeans        = 0.00016f;
    float lrScales       = 0.005f;
    float lrQuats        = 0.001f;
    float lrFeaturesDc   = 0.0025f;
    float lrFeaturesRest = 0.000125f;
    float lrOpacities    = 0.05f;
    float lambdaRgb      = 0.8f;
    float lambdaSsim     = 0.2f;
    float lambdaDepth    = 0.5f;
    float lambdaReg      = 0.001f;
    /// Wall-clock budget in milliseconds per keyframe for online updates.
    double budgetMs       = 30.0;
};

/// Runs a small number of local optimisation steps on a subset of Gaussians.
///
/// Maintains Adam optimizer state only for the unstable subset to keep
/// memory bounded.  The subset changes between keyframes; on each call the
/// optimizer is rebuilt for the new candidate set.
class LocalOptimizer {
public:
    explicit LocalOptimizer(const OptimizerConfig &cfg,
                            const torch::Device &device);
    ~LocalOptimizer();

    struct RenderInput {
        // [H,W,3] float32 observed RGB [0,1]
        torch::Tensor obsRgb;
        // [H,W]   float32 observed depth in metres
        torch::Tensor obsDepth;
        float fx, fy, cx, cy;
        torch::Tensor camToWorld;   // 4×4 float32
    };

    /// Online update: runs up to cfg.onlineSteps, stopping early if budget
    /// is exceeded.  candidateIndices is the set from GaussianScheduler.
    void optimizeOnline(LiveGaussianMap &map,
                        const torch::Tensor &candidateIndices,
                        const RenderInput &render);

    /// Background update: runs cfg.backgroundSteps without a wall-clock
    /// budget.  Intended to be called from a separate thread.
    void optimizeBackground(LiveGaussianMap &map,
                            const torch::Tensor &candidateIndices,
                            const std::vector<RenderInput> &keyframeBatch);

    const OptimizerConfig& config() const { return cfg_; }

private:
    void rebuildOptimizers(LiveGaussianMap &map,
                           const torch::Tensor &candidateIndices);
    void releaseOptimizers();

    torch::Tensor renderAndLoss(LiveGaussianMap &map,
                                const torch::Tensor &candidateIndices,
                                const RenderInput &render,
                                float &outLossRgb,
                                float &outLossDepth);

    OptimizerConfig cfg_;
    torch::Device   device_;

    // Per-parameter Adam optimizers (rebuilt each keyframe)
    torch::optim::Adam *meansOpt_       = nullptr;
    torch::optim::Adam *scalesOpt_      = nullptr;
    torch::optim::Adam *quatsOpt_       = nullptr;
    torch::optim::Adam *featuresDcOpt_  = nullptr;
    torch::optim::Adam *opacitiesOpt_   = nullptr;
};

} // namespace opensplat_live
