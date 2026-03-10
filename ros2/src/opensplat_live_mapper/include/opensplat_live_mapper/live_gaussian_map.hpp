#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <torch/torch.h>

namespace opensplat_live {

/// Per-Gaussian live bookkeeping that augments the raw tensors.
struct GaussianMeta {
    int32_t  bucket_id          = -1;
    int32_t  age_frames         = 0;
    int32_t  last_seen_frame    = -1;
    int32_t  visibility_count   = 0;
    float    confidence         = 0.0f;
    float    mean_color_residual= 0.0f;
    float    mean_depth_residual= 0.0f;
    int32_t  recent_hit_count   = 0;
    float    recent_gradient_norm = 0.0f;
    bool     stable_flag        = false;
    bool     dynamic_suspect    = false;
    bool     active_flag        = false;
    float    refine_priority    = 0.0f;
    int32_t  frozen_until_frame = -1;
    int32_t  local_window_id    = -1;
};

/// Central Gaussian map – owns all parameter tensors and metadata.
///
/// All tensor operations are protected by a shared mutex; callers that need
/// consistent snapshots across multiple accessors should hold the lock
/// themselves via lockGuard().
class LiveGaussianMap {
public:
    explicit LiveGaussianMap(const torch::Device &device,
                             int maxCapacity = 2'000'000);
    ~LiveGaussianMap() = default;

    // ── Capacity / size ───────────────────────────────────────────────────
    int64_t size() const;
    int64_t capacity() const { return maxCapacity_; }

    // ── Bulk append ───────────────────────────────────────────────────────
    /// Append a batch of new Gaussians.  All tensors must have the same
    /// leading dimension.  Returns the index range [first, last).
    std::pair<int64_t,int64_t> append(
        const torch::Tensor &means,       // [N,3] float32
        const torch::Tensor &scales,      // [N,3] float32 (log-space)
        const torch::Tensor &quats,       // [N,4] float32
        const torch::Tensor &featuresDc,  // [N,3] float32
        const torch::Tensor &featuresRest,// [N,(D-1)*3] float32
        const torch::Tensor &opacities,   // [N,1] float32 (logit-space)
        const torch::Tensor &normals,     // [N,3] float32
        const std::vector<GaussianMeta> &meta);

    // ── Remove ────────────────────────────────────────────────────────────
    /// Remove Gaussians at indices marked true in deletedMask [N] bool.
    void remove(const torch::Tensor &deletedMask);

    // ── Parameter accessors (no copy – returns view) ──────────────────────
    torch::Tensor& means()        { return means_; }
    torch::Tensor& scales()       { return scales_; }
    torch::Tensor& quats()        { return quats_; }
    torch::Tensor& featuresDc()   { return featuresDc_; }
    torch::Tensor& featuresRest() { return featuresRest_; }
    torch::Tensor& opacities()    { return opacities_; }
    torch::Tensor& normals()      { return normals_; }

    const torch::Tensor& means()        const { return means_; }
    const torch::Tensor& scales()       const { return scales_; }
    const torch::Tensor& quats()        const { return quats_; }
    const torch::Tensor& featuresDc()   const { return featuresDc_; }
    const torch::Tensor& featuresRest() const { return featuresRest_; }
    const torch::Tensor& opacities()    const { return opacities_; }
    const torch::Tensor& normals()      const { return normals_; }

    // ── Metadata accessors ────────────────────────────────────────────────
    std::vector<GaussianMeta>&       meta()       { return meta_; }
    const std::vector<GaussianMeta>& meta() const { return meta_; }

    // ── Stable / unstable masks (GPU tensors, refreshed by Scheduler) ─────
    /// Boolean tensor [N] – true for stable Gaussians.
    torch::Tensor stableMask() const;
    /// Boolean tensor [N] – true for Gaussians eligible for optimization.
    torch::Tensor unstableMask() const;
    /// Indices of active Gaussians (union of active buckets).
    torch::Tensor activeIndices() const;

    // ── Frame counter ─────────────────────────────────────────────────────
    int32_t currentFrame() const { return currentFrame_; }
    void incrementFrame() { ++currentFrame_; }

    // ── Thread safety ─────────────────────────────────────────────────────
    std::unique_lock<std::mutex> lockGuard() {
        return std::unique_lock<std::mutex>(mutex_);
    }

    // ── Device ───────────────────────────────────────────────────────────
    const torch::Device& device() const { return device_; }

    // ── Stable / unstable masks (raw bool tensors kept in sync by
    //    GaussianScheduler and updated after every insert/remove) ──────────
    torch::Tensor stableTensor_;    // [N] bool  (CPU or GPU)
    torch::Tensor unstableTensor_;  // [N] bool

private:
    void growIfNeeded(int64_t extra);

    torch::Device device_;
    int64_t       maxCapacity_;
    int64_t       n_ = 0;           // current count

    // ── Core parameter tensors ─────────────────────────────────────────
    torch::Tensor means_;
    torch::Tensor scales_;
    torch::Tensor quats_;
    torch::Tensor featuresDc_;
    torch::Tensor featuresRest_;
    torch::Tensor opacities_;
    torch::Tensor normals_;

    std::vector<GaussianMeta> meta_;

    int32_t currentFrame_ = 0;
    mutable std::mutex mutex_;
};

} // namespace opensplat_live
