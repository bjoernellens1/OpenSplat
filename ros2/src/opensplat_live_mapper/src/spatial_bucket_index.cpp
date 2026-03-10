#include "opensplat_live_mapper/spatial_bucket_index.hpp"
#include <cmath>
#include <stdexcept>

namespace opensplat_live {

const std::unordered_set<int64_t> SpatialBucketIndex::emptySet_{};

SpatialBucketIndex::SpatialBucketIndex(float blockSize)
    : blockSize_(blockSize) {}

SpatialBucketIndex::VoxelKey
SpatialBucketIndex::keyOf(const std::array<float,3> &xyz) const {
    return {
        static_cast<int32_t>(std::floor(xyz[0] / blockSize_)),
        static_cast<int32_t>(std::floor(xyz[1] / blockSize_)),
        static_cast<int32_t>(std::floor(xyz[2] / blockSize_))
    };
}

BucketId SpatialBucketIndex::getOrCreate(const VoxelKey &key) {
    auto it = keyToId_.find(key);
    if (it != keyToId_.end()) return it->second;
    BucketId id = static_cast<BucketId>(idToKey_.size());
    keyToId_[key] = id;
    idToKey_.push_back(key);
    return id;
}

BucketId SpatialBucketIndex::bucketOf(const std::array<float,3> &xyz) const {
    auto key = keyOf(xyz);
    auto it = keyToId_.find(key);
    return it != keyToId_.end() ? it->second : INVALID_BUCKET;
}

void SpatialBucketIndex::insert(int64_t gaussianIdx,
                                const std::array<float,3> &xyz) {
    BucketId id = getOrCreate(keyOf(xyz));
    buckets_[id].insert(gaussianIdx);
}

void SpatialBucketIndex::remove(int64_t gaussianIdx,
                                const std::array<float,3> &xyz) {
    auto key = keyOf(xyz);
    auto kit = keyToId_.find(key);
    if (kit == keyToId_.end()) return;
    auto bit = buckets_.find(kit->second);
    if (bit != buckets_.end()) bit->second.erase(gaussianIdx);
}

void SpatialBucketIndex::update(int64_t gaussianIdx,
                                const std::array<float,3> &oldXyz,
                                const std::array<float,3> &newXyz) {
    remove(gaussianIdx, oldXyz);
    insert(gaussianIdx, newXyz);
}

const std::unordered_set<int64_t>&
SpatialBucketIndex::indicesInBucket(BucketId id) const {
    auto it = buckets_.find(id);
    return it != buckets_.end() ? it->second : emptySet_;
}

std::vector<BucketId>
SpatialBucketIndex::queryAABB(const std::array<float,3> &minXyz,
                               const std::array<float,3> &maxXyz) const {
    VoxelKey kMin = keyOf(minXyz);
    VoxelKey kMax = keyOf(maxXyz);
    std::vector<BucketId> result;
    for (int32_t x = kMin.x; x <= kMax.x; ++x)
    for (int32_t y = kMin.y; y <= kMax.y; ++y)
    for (int32_t z = kMin.z; z <= kMax.z; ++z) {
        auto it = keyToId_.find({x, y, z});
        if (it != keyToId_.end()) result.push_back(it->second);
    }
    return result;
}

std::vector<BucketId>
SpatialBucketIndex::queryFrustum(const float *c2w,
                                  float fx, float fy,
                                  float cx, float cy,
                                  int width, int height,
                                  float zNear, float zFar) const {
    // Build world-space frustum planes and test voxel-centre points.
    // For each populated bucket, test if its voxel centre is visible.
    std::vector<BucketId> result;
    result.reserve(64);

    // World-to-camera: inverse of c2w (assumes orthonormal upper-left 3×3)
    // R^T · (p - t)
    float r00=c2w[0],r01=c2w[1],r02=c2w[2], tx=c2w[3];
    float r10=c2w[4],r11=c2w[5],r12=c2w[6], ty=c2w[7];
    float r20=c2w[8],r21=c2w[9],r22=c2w[10],tz=c2w[11];

    for (auto &[key, id] : keyToId_) {
        if (buckets_.count(id) == 0 || buckets_.at(id).empty()) continue;

        // Voxel centre in world space
        float wx = (key.x + 0.5f) * blockSize_;
        float wy = (key.y + 0.5f) * blockSize_;
        float wz = (key.z + 0.5f) * blockSize_;

        // Transform to camera space (R^T * (p - t))
        float dx = wx - tx, dy = wy - ty, dz = wz - tz;
        float pcx = r00*dx + r10*dy + r20*dz;
        float pcy = r01*dx + r11*dy + r21*dz;
        float pcz = r02*dx + r12*dy + r22*dz;

        if (pcz < zNear || pcz > zFar) continue;

        // Project to image
        float u = fx * pcx / pcz + cx;
        float v = fy * pcy / pcz + cy;

        float margin = blockSize_ * fx / pcz; // one-bucket margin in pixels
        if (u + margin < 0 || u - margin > width)  continue;
        if (v + margin < 0 || v - margin > height) continue;

        result.push_back(id);
    }
    return result;
}

std::vector<BucketId> SpatialBucketIndex::neighbours(BucketId id) const {
    if (static_cast<size_t>(id) >= idToKey_.size()) return {};
    const VoxelKey &k = idToKey_[id];
    std::vector<BucketId> result;
    result.reserve(27);
    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        VoxelKey nk{k.x+dx, k.y+dy, k.z+dz};
        auto it = keyToId_.find(nk);
        if (it != keyToId_.end()) result.push_back(it->second);
    }
    return result;
}

void SpatialBucketIndex::markActive(BucketId id, int32_t frameIdx) {
    activeSince_[id] = frameIdx;
}

std::vector<BucketId> SpatialBucketIndex::activeBuckets() const {
    std::vector<BucketId> result;
    result.reserve(activeSince_.size());
    for (auto &[id, _] : activeSince_) result.push_back(id);
    return result;
}

void SpatialBucketIndex::clearActive() { activeSince_.clear(); }

size_t SpatialBucketIndex::bucketSize(BucketId id) const {
    auto it = buckets_.find(id);
    return it != buckets_.end() ? it->second.size() : 0u;
}

} // namespace opensplat_live
