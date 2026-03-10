#include "opensplat_live_mapper/live_gaussian_map.hpp"
#include <stdexcept>

using namespace torch::indexing;

namespace opensplat_live {

static constexpr int SH_REST_BASES = 3; // (degree-1 + 1)^2 - 1 = 3 for SH degree 1

LiveGaussianMap::LiveGaussianMap(const torch::Device &device, int maxCapacity)
    : device_(device), maxCapacity_(maxCapacity), n_(0) {
    // Pre-allocate empty tensors; append() will grow them lazily.
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    means_        = torch::empty({0, 3},  opts);
    scales_       = torch::empty({0, 3},  opts);
    quats_        = torch::empty({0, 4},  opts);
    featuresDc_   = torch::empty({0, 3},  opts);
    featuresRest_ = torch::empty({0, SH_REST_BASES, 3}, opts);
    opacities_    = torch::empty({0, 1},  opts);
    normals_      = torch::empty({0, 3},  opts);

    auto boolOpts = torch::TensorOptions().dtype(torch::kBool).device(device_);
    stableTensor_   = torch::empty({0}, boolOpts);
    unstableTensor_ = torch::empty({0}, boolOpts);
}

std::pair<int64_t,int64_t> LiveGaussianMap::append(
    const torch::Tensor &means,
    const torch::Tensor &scales,
    const torch::Tensor &quats,
    const torch::Tensor &featuresDc,
    const torch::Tensor &featuresRest,
    const torch::Tensor &opacities,
    const torch::Tensor &normals,
    const std::vector<GaussianMeta> &meta) {

    const int64_t N = means.size(0);
    if (N == 0) return {n_, n_};
    if (n_ + N > maxCapacity_)
        throw std::runtime_error("LiveGaussianMap: capacity exceeded");

    auto toDevice = [&](const torch::Tensor &t) {
        return t.to(device_).detach();
    };

    means_        = torch::cat({means_,        toDevice(means)},        0);
    scales_       = torch::cat({scales_,       toDevice(scales)},       0);
    quats_        = torch::cat({quats_,        toDevice(quats)},        0);
    featuresDc_   = torch::cat({featuresDc_,   toDevice(featuresDc)},   0);
    featuresRest_ = torch::cat({featuresRest_, toDevice(featuresRest)}, 0);
    opacities_    = torch::cat({opacities_,    toDevice(opacities)},    0);
    normals_      = torch::cat({normals_,      toDevice(normals)},      0);

    // Attach grad requirements
    means_.requires_grad_(true);
    scales_.requires_grad_(true);
    quats_.requires_grad_(true);
    featuresDc_.requires_grad_(true);
    featuresRest_.requires_grad_(true);
    opacities_.requires_grad_(true);

    // Expand boolean mask tensors
    auto boolOpts = torch::TensorOptions().dtype(torch::kBool).device(device_);
    auto newFalse  = torch::zeros({N}, boolOpts);
    auto newTrue   = torch::ones( {N}, boolOpts);
    stableTensor_   = torch::cat({stableTensor_,   newFalse}, 0);
    unstableTensor_ = torch::cat({unstableTensor_, newTrue},  0);  // new = unstable

    // Metadata
    meta_.insert(meta_.end(), meta.begin(), meta.end());

    int64_t first = n_;
    n_ += N;
    return {first, n_};
}

void LiveGaussianMap::remove(const torch::Tensor &deletedMask) {
    // deletedMask: [N] bool tensor (true = remove)
    torch::Tensor keepMask = ~deletedMask.to(device_);

    auto keep = [&](const torch::Tensor &t) {
        return t.index({keepMask}).detach().requires_grad_(t.requires_grad());
    };

    means_        = keep(means_);
    scales_       = keep(scales_);
    quats_        = keep(quats_);
    featuresDc_   = keep(featuresDc_);
    featuresRest_ = keep(featuresRest_);
    opacities_    = keep(opacities_);
    normals_      = keep(normals_);
    stableTensor_   = stableTensor_.index({keepMask});
    unstableTensor_ = unstableTensor_.index({keepMask});

    // Update metadata
    auto keepCpu = keepMask.to(torch::kCPU);
    auto keepAcc = keepCpu.accessor<bool,1>();
    std::vector<GaussianMeta> newMeta;
    newMeta.reserve(static_cast<size_t>(keepMask.sum().item<int64_t>()));
    for (int64_t i = 0; i < static_cast<int64_t>(meta_.size()); ++i) {
        if (keepAcc[i]) newMeta.push_back(meta_[i]);
    }
    meta_ = std::move(newMeta);
    n_ = means_.size(0);
}

int64_t LiveGaussianMap::size() const { return n_; }

torch::Tensor LiveGaussianMap::stableMask() const { return stableTensor_; }
torch::Tensor LiveGaussianMap::unstableMask() const { return unstableTensor_; }

torch::Tensor LiveGaussianMap::activeIndices() const {
    return unstableTensor_.nonzero().squeeze(1);
}

} // namespace opensplat_live
