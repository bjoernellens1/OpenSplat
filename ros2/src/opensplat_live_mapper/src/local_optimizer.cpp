#include "opensplat_live_mapper/local_optimizer.hpp"
#include <chrono>
#include <iostream>

namespace opensplat_live {

/// SH DC coefficient to RGB conversion: rgb = SH_C0 * sh_dc + 0.5
/// Inverse: sh_dc = (rgb - 0.5) / SH_C0
/// SH_C0 = 1 / (2 * sqrt(pi)) ≈ 0.28209479177387814
static constexpr float SH_C0 = 0.28209479177387814f;

LocalOptimizer::LocalOptimizer(const OptimizerConfig &cfg,
                               const torch::Device &device)
    : cfg_(cfg), device_(device) {}

LocalOptimizer::~LocalOptimizer() { releaseOptimizers(); }

void LocalOptimizer::releaseOptimizers() {
    delete meansOpt_;       meansOpt_      = nullptr;
    delete scalesOpt_;      scalesOpt_     = nullptr;
    delete quatsOpt_;       quatsOpt_      = nullptr;
    delete featuresDcOpt_;  featuresDcOpt_ = nullptr;
    delete opacitiesOpt_;   opacitiesOpt_  = nullptr;
}

void LocalOptimizer::rebuildOptimizers(LiveGaussianMap &map,
                                        const torch::Tensor &candidateIndices) {
    releaseOptimizers();
    if (candidateIndices.numel() == 0) return;

    // Slice views for the candidate subset; these are the parameters
    // that will be updated.  We do NOT build optimizers for the full map.
    using namespace torch::indexing;
    auto idx = candidateIndices.to(device_);

    // Ensure grad is enabled on the full tensors (they retain grad by default
    // after LiveGaussianMap::append).  The Adam optimizers will accumulate
    // gradients only at the candidate rows via the indexed backward pass.
    meansOpt_      = new torch::optim::Adam({map.means()},
                         torch::optim::AdamOptions(cfg_.lrMeans));
    scalesOpt_     = new torch::optim::Adam({map.scales()},
                         torch::optim::AdamOptions(cfg_.lrScales));
    quatsOpt_      = new torch::optim::Adam({map.quats()},
                         torch::optim::AdamOptions(cfg_.lrQuats));
    featuresDcOpt_ = new torch::optim::Adam({map.featuresDc()},
                         torch::optim::AdamOptions(cfg_.lrFeaturesDc));
    opacitiesOpt_  = new torch::optim::Adam({map.opacities()},
                         torch::optim::AdamOptions(cfg_.lrOpacities));
}

// Minimal alpha-composite depth render using Gaussian opacities and means.
// For v1 this is a simplified soft-depth render rather than a full splat.
torch::Tensor LocalOptimizer::renderAndLoss(LiveGaussianMap &map,
                                             const torch::Tensor &candidateIndices,
                                             const RenderInput &render,
                                             float &outLossRgb,
                                             float &outLossDepth) {
    using namespace torch::indexing;
    // For v1 we compute a simple photometric + depth L1 over candidate points.
    // A full differentiable splat render would replace this in production.

    auto idx = candidateIndices.to(device_);
    if (idx.numel() == 0) {
        outLossRgb   = 0.0f;
        outLossDepth = 0.0f;
        return torch::zeros({1}, torch::TensorOptions()
                            .dtype(torch::kFloat32).device(device_)
                            .requires_grad(true));
    }

    // Project candidate Gaussian means to image plane
    auto means = map.means().index({idx}).to(device_);       // [M,3]
    auto c2wInv = render.camToWorld.to(device_).inverse();   // [4,4]
    auto R = c2wInv.slice(0,0,3).slice(1,0,3);              // [3,3]
    auto t = c2wInv.slice(0,0,3).slice(1,3,4);              // [3,1]
    auto camPts = torch::mm(means, R.transpose(0,1))
                + t.transpose(0,1).expand({means.size(0), 3}); // [M,3]
    auto pz = camPts.index({Slice(), 2});                     // [M]
    auto validZ = pz > 0.01f;
    auto px = camPts.index({Slice(),0}) / pz * render.fx + render.cx;
    auto py = camPts.index({Slice(),1}) / pz * render.fy + render.cy;

    int H = render.obsRgb.size(0), W = render.obsRgb.size(1);
    auto inFrustum = validZ
                   & (px >= 0) & (px < W-1)
                   & (py >= 0) & (py < H-1);
    if (inFrustum.sum().item<int64_t>() == 0) {
        outLossRgb   = 0.0f;
        outLossDepth = 0.0f;
        return torch::zeros({1}, torch::TensorOptions()
                            .dtype(torch::kFloat32).device(device_)
                            .requires_grad(true));
    }

    auto pxV = px.index({inFrustum}).to(torch::kLong).clamp(0, W-1);
    auto pyV = py.index({inFrustum}).to(torch::kLong).clamp(0, H-1);

    auto obsRgbDev  = render.obsRgb.to(device_);   // [H,W,3]
    auto obsDepDev  = render.obsDepth.to(device_); // [H,W]
    auto dc = map.featuresDc().index({idx}).index({inFrustum}).to(device_); // [M,3]
    auto predRgb = (dc * SH_C0 + 0.5f).clamp(0.0f, 1.0f); // [M,3]
    auto gtRgb   = obsRgbDev.index({pyV, pxV}); // [M,3]

    auto predZ   = pz.index({inFrustum});               // [M]
    auto gtDepth = obsDepDev.index({pyV, pxV});         // [M]
    auto validDepth = gtDepth > 0.01f;

    torch::Tensor lossRgb   = (predRgb - gtRgb).abs().mean();
    torch::Tensor lossDepth = torch::zeros({1}, predRgb.options());
    if (validDepth.sum().item<int64_t>() > 0) {
        lossDepth = (predZ.index({validDepth}) - gtDepth.index({validDepth})).abs().mean();
    }

    outLossRgb   = lossRgb.item<float>();
    outLossDepth = lossDepth.item<float>();

    return cfg_.lambdaRgb * lossRgb + cfg_.lambdaDepth * lossDepth;
}

void LocalOptimizer::optimizeOnline(LiveGaussianMap &map,
                                     const torch::Tensor &candidateIndices,
                                     const RenderInput &render) {
    if (candidateIndices.numel() == 0) return;

    rebuildOptimizers(map, candidateIndices);
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::duration<double, std::milli>(cfg_.budgetMs);

    for (int step = 0; step < cfg_.onlineSteps; ++step) {
        if (std::chrono::steady_clock::now() >= deadline) break;

        meansOpt_->zero_grad();
        scalesOpt_->zero_grad();
        quatsOpt_->zero_grad();
        featuresDcOpt_->zero_grad();
        opacitiesOpt_->zero_grad();

        float lRgb = 0.0f, lDepth = 0.0f;
        auto loss = renderAndLoss(map, candidateIndices, render, lRgb, lDepth);
        if (loss.item<float>() < 1e-9f) break;

        loss.backward();
        meansOpt_->step();
        scalesOpt_->step();
        quatsOpt_->step();
        featuresDcOpt_->step();
        opacitiesOpt_->step();
    }
    releaseOptimizers();
}

void LocalOptimizer::optimizeBackground(
        LiveGaussianMap &map,
        const torch::Tensor &candidateIndices,
        const std::vector<RenderInput> &keyframeBatch) {
    if (candidateIndices.numel() == 0 || keyframeBatch.empty()) return;

    rebuildOptimizers(map, candidateIndices);

    for (int step = 0; step < cfg_.backgroundSteps; ++step) {
        for (const auto &render : keyframeBatch) {
            meansOpt_->zero_grad();
            scalesOpt_->zero_grad();
            quatsOpt_->zero_grad();
            featuresDcOpt_->zero_grad();
            opacitiesOpt_->zero_grad();

            float lRgb = 0.0f, lDepth = 0.0f;
            auto loss = renderAndLoss(map, candidateIndices, render, lRgb, lDepth);
            if (loss.item<float>() < 1e-9f) continue;

            loss.backward();
            meansOpt_->step();
            scalesOpt_->step();
            quatsOpt_->step();
            featuresDcOpt_->step();
            opacitiesOpt_->step();
        }
    }
    releaseOptimizers();
}

} // namespace opensplat_live
