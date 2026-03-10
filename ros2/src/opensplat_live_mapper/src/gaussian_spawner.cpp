#include "opensplat_live_mapper/gaussian_spawner.hpp"
#include <cmath>

namespace opensplat_live {

// SH_C0 = 1 / (2*sqrt(pi)) ≈ 0.28209479177387814
static constexpr float SH_C0 = 0.28209479177387814f;

// Number of SH bases for a given degree
static int numShBases(int deg) { return (deg+1)*(deg+1); }

GaussianSpawner::GaussianSpawner(const SpawnerConfig &cfg) : cfg_(cfg) {}

// Estimate per-pixel surface normals from the depth image via central differences.
torch::Tensor GaussianSpawner::estimateNormals(const torch::Tensor &depth,
                                               float fx, float fy,
                                               float cx, float cy) {
    int H = depth.size(0), W = depth.size(1);
    // Backproject central, right, and down neighbours
    // dz_dx and dz_dy give tangent vectors; cross product gives normal
    using namespace torch::indexing;
    auto d  = depth.unsqueeze(2).to(torch::kFloat32);  // [H,W,1]

    // pixel grids
    auto ys = torch::arange(H, depth.options().dtype(torch::kFloat32));
    auto xs = torch::arange(W, depth.options().dtype(torch::kFloat32));
    // [H,W]
    auto gridY = ys.unsqueeze(1).expand({H,W});
    auto gridX = xs.unsqueeze(0).expand({H,W});

    // 3-D coordinates [H,W,3]
    auto px = (gridX - cx) / fx;
    auto py = (gridY - cy) / fy;
    auto pts = torch::stack({px * depth, py * depth, depth}, 2); // [H,W,3]

    // Pad before shifting
    auto pts_pad = torch::nn::functional::pad(pts,
        torch::nn::functional::PadFuncOptions({0,0,0,1,0,1}).mode(torch::kReplicate));

    // Tangent along x and y
    auto tx = pts_pad.index({Slice(0,H), Slice(1,W+1), Slice()})
            - pts_pad.index({Slice(0,H), Slice(0,W),   Slice()});
    auto ty = pts_pad.index({Slice(1,H+1), Slice(0,W), Slice()})
            - pts_pad.index({Slice(0,H),   Slice(0,W), Slice()});

    // Cross product tx × ty
    auto nx = tx.index({Slice(),Slice(),1}) * ty.index({Slice(),Slice(),2})
            - tx.index({Slice(),Slice(),2}) * ty.index({Slice(),Slice(),1});
    auto ny = tx.index({Slice(),Slice(),2}) * ty.index({Slice(),Slice(),0})
            - tx.index({Slice(),Slice(),0}) * ty.index({Slice(),Slice(),2});
    auto nz = tx.index({Slice(),Slice(),0}) * ty.index({Slice(),Slice(),1})
            - tx.index({Slice(),Slice(),1}) * ty.index({Slice(),Slice(),0});

    auto normals = torch::stack({nx,ny,nz}, 2); // [H,W,3]
    auto norm    = normals.norm(2, 2, /*keepdim=*/true).clamp_min(1e-6f);
    return normals / norm;
}

// Backproject selected pixels to world 3-D using cam-to-world matrix.
torch::Tensor GaussianSpawner::backproject(const torch::Tensor &depth,
                                            const torch::Tensor &mask,
                                            float fx, float fy,
                                            float cx, float cy,
                                            const torch::Tensor &camToWorld) {
    // mask: [H,W] bool; depth: [H,W] float32; camToWorld: [4,4] float32 CPU
    using namespace torch::indexing;
    int H = depth.size(0), W = depth.size(1);
    auto ys = torch::arange(H, depth.options().dtype(torch::kFloat32));
    auto xs = torch::arange(W, depth.options().dtype(torch::kFloat32));
    auto gridY = ys.unsqueeze(1).expand({H,W});
    auto gridX = xs.unsqueeze(0).expand({H,W});

    auto px = ((gridX - cx) / fx) * depth;
    auto py = ((gridY - cy) / fy) * depth;
    auto pz = depth;
    auto ones = torch::ones_like(pz);

    // Stack [H*W, 4]
    auto pts = torch::stack({px.flatten(), py.flatten(), pz.flatten(), ones.flatten()}, 1);
    auto maskFlat = mask.flatten();
    pts = pts.index({maskFlat});  // [M,4]

    // Transform: [M,4] @ [4,4]^T → world coords [M,4]
    auto c2w = camToWorld.to(depth.device()).to(torch::kFloat32);
    auto worldPts = torch::mm(pts, c2w.transpose(0,1)); // [M,4]
    return worldPts.index({Slice(), Slice(0,3)});        // [M,3]
}

GaussianSpawner::Output
GaussianSpawner::spawn(const Input &in,
                       const LiveGaussianMap &map,
                       SpatialBucketIndex &index) {
    using namespace torch::indexing;
    Output out;

    torch::Tensor depth = in.depthImage.to(torch::kFloat32);
    torch::Tensor rgb   = in.rgbImage.to(torch::kFloat32);
    int H = depth.size(0), W = depth.size(1);

    // ── Valid depth mask ───────────────────────────────────────────────────
    torch::Tensor validMask = (depth > cfg_.minDepth) & (depth < cfg_.maxDepth);

    // ── Class 1: newly observed (no nearby Gaussian) ──────────────────────
    // We use a uniform sub-sampling of valid pixels as a proxy.
    torch::Tensor class1 = validMask.clone();

    // ── Class 2: high colour residual ─────────────────────────────────────
    torch::Tensor class2 = torch::zeros({H, W}, torch::kBool);
    if (in.colorResidual.defined() && in.colorResidual.numel() > 0) {
        auto resid = in.colorResidual.norm(2, 2); // [H,W]
        class2 = validMask & (resid > cfg_.colorResidualThr);
    }

    // ── Class 3: high depth residual ─────────────────────────────────────
    torch::Tensor class3 = torch::zeros({H, W}, torch::kBool);
    if (in.depthResidual.defined() && in.depthResidual.numel() > 0) {
        class3 = validMask & (in.depthResidual.abs() > cfg_.depthResidualThr);
    }

    // Union of candidates
    torch::Tensor candidateMask = class1 | class2 | class3;

    // ── Sub-sample to max budget ──────────────────────────────────────────
    torch::Tensor candIdx = candidateMask.nonzero(); // [M,2]
    int64_t M = candIdx.size(0);
    if (M == 0) return out;

    if (M > cfg_.maxPerKeyframe) {
        auto perm = torch::randperm(M, torch::kLong).slice(0, 0, cfg_.maxPerKeyframe);
        candIdx = candIdx.index({perm});
        M = cfg_.maxPerKeyframe;
    }

    // ── Build per-pixel masks from sampled indices ────────────────────────
    torch::Tensor sampledMask = torch::zeros({H, W}, torch::kBool);
    sampledMask.index_put_({candIdx.index({Slice(), 0}),
                             candIdx.index({Slice(), 1})},
                            torch::ones({M}, torch::kBool));

    // ── Backproject to 3-D ───────────────────────────────────────────────
    auto meansXyz = backproject(depth, sampledMask, in.fx, in.fy, in.cx, in.cy,
                                 in.camToWorld); // [N,3]
    int64_t N = meansXyz.size(0);
    if (N == 0) return out;

    // ── Estimate normals ─────────────────────────────────────────────────
    auto localNormals = estimateNormals(depth, in.fx, in.fy, in.cx, in.cy); // [H,W,3]
    auto normFlat = localNormals.reshape({H*W, 3});
    auto maskFlat = sampledMask.flatten();
    auto spawnedNormals = normFlat.index({maskFlat}); // [N,3]
    // Rotate normals to world frame (upper-left 3×3 of camToWorld)
    auto R = in.camToWorld.to(torch::kFloat32).slice(0, 0, 3).slice(1, 0, 3); // [3,3]
    spawnedNormals = torch::mm(spawnedNormals.to(R.device()), R.transpose(0,1));

    // ── Colours for featuresDc ────────────────────────────────────────────
    auto rgbFlat = rgb.reshape({H*W, 3});
    auto spawnedRgb = rgbFlat.index({maskFlat}); // [N,3] in [0,1]
    // Convert to first SH coefficient: sh_dc = (rgb - 0.5) / SH_C0
    auto featuresDc = (spawnedRgb - 0.5f) / SH_C0; // [N,3]

    // ── Scales from depth ─────────────────────────────────────────────────
    auto depthFlat  = depth.flatten();
    auto spawnedDepth = depthFlat.index({maskFlat});           // [N]
    auto scale1D      = (spawnedDepth * cfg_.initScaleFromDepth).clamp_min(1e-4f);
    // Anisotropic: smaller along normal, larger in tangent plane
    auto scaleXyz = torch::stack({scale1D, scale1D, scale1D * cfg_.normalScaleFactor}, 1).log(); // [N,3]

    // ── Quaternions aligned to normals ────────────────────────────────────
    // Simple: quats = [1,0,0,0] (identity) – scheduler will refine
    auto quats = torch::zeros({N, 4}, torch::kFloat32);
    quats.index_put_({Slice(), 0}, 1.0f);

    // ── Opacities ────────────────────────────────────────────────────────
    float logitInit = std::log(cfg_.initOpacity / (1.0f - cfg_.initOpacity));
    auto opacities = torch::full({N, 1}, logitInit, torch::kFloat32);

    // ── Higher-order SH (zero-initialised) ───────────────────────────────
    int shBases = numShBases(cfg_.shDegree);
    auto featuresRest = torch::zeros({N, std::max(shBases - 1, 0), 3},
                                     torch::kFloat32);

    // ── Metadata ─────────────────────────────────────────────────────────
    std::vector<GaussianMeta> meta(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        meta[i].age_frames      = 0;
        meta[i].last_seen_frame = in.frameIdx;
        meta[i].stable_flag     = false;
        meta[i].active_flag     = true;
        meta[i].confidence      = 0.0f;
        meta[i].refine_priority = 1.0f;
    }

    out.means        = meansXyz.cpu();
    out.scales       = scaleXyz.cpu();
    out.quats        = quats.cpu();
    out.featuresDc   = featuresDc.cpu();
    out.featuresRest = featuresRest.cpu();
    out.opacities    = opacities.cpu();
    out.normals      = spawnedNormals.cpu();
    out.meta         = std::move(meta);
    return out;
}

} // namespace opensplat_live
