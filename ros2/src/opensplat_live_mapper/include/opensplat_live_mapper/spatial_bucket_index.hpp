#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace opensplat_live {

using BucketId = int32_t;
static constexpr BucketId INVALID_BUCKET = -1;

/// Voxel-hashed spatial index over Gaussian centroids.
///
/// The world is subdivided into axis-aligned cubic buckets of size
/// `blockSize` metres.  Each bucket stores the indices of the Gaussians
/// whose mean falls inside it.
///
/// This class is NOT thread-safe.  External locking (e.g. from
/// LiveGaussianMap::lockGuard) is expected.
class SpatialBucketIndex {
public:
    explicit SpatialBucketIndex(float blockSize = 0.30f);

    // ── Insertion / removal ───────────────────────────────────────────────
    void insert(int64_t gaussianIdx, const std::array<float,3> &xyz);
    void remove(int64_t gaussianIdx, const std::array<float,3> &xyz);
    /// Update position of an existing Gaussian (remove old, insert new).
    void update(int64_t gaussianIdx,
                const std::array<float,3> &oldXyz,
                const std::array<float,3> &newXyz);

    // ── Queries ───────────────────────────────────────────────────────────
    /// All Gaussian indices in bucket `id`.
    const std::unordered_set<int64_t>& indicesInBucket(BucketId id) const;

    /// Return bucket ids that overlap the given axis-aligned bounding box.
    std::vector<BucketId> queryAABB(const std::array<float,3> &minXyz,
                                    const std::array<float,3> &maxXyz) const;

    /// Return bucket ids whose voxel centres fall inside the camera frustum
    /// defined by the 4×4 camera-to-world matrix and projection parameters.
    std::vector<BucketId> queryFrustum(
        const float *camToWorld4x4,   // row-major 4×4
        float fx, float fy, float cx, float cy,
        int width, int height,
        float zNear = 0.1f, float zFar = 10.0f) const;

    /// 26-connected neighbours of `id` (including `id` itself).
    std::vector<BucketId> neighbours(BucketId id) const;

    // ── Bookkeeping ───────────────────────────────────────────────────────
    /// Mark a bucket as active (touched by current frame).
    void markActive(BucketId id, int32_t frameIdx);
    /// Return ids of all currently active buckets.
    std::vector<BucketId> activeBuckets() const;
    /// Clear activity flags (call at start of each frame).
    void clearActive();

    /// Number of Gaussians in a bucket.
    size_t bucketSize(BucketId id) const;

    /// Total number of populated buckets.
    size_t numBuckets() const { return buckets_.size(); }

    float blockSize() const { return blockSize_; }

    BucketId bucketOf(const std::array<float,3> &xyz) const;

private:
    struct VoxelKey {
        int32_t x, y, z;
        bool operator==(const VoxelKey &o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct VoxelKeyHash {
        size_t operator()(const VoxelKey &k) const noexcept {
            // FNV-style combine
            size_t h = 2166136261u;
            h ^= static_cast<uint32_t>(k.x); h *= 16777619u;
            h ^= static_cast<uint32_t>(k.y); h *= 16777619u;
            h ^= static_cast<uint32_t>(k.z); h *= 16777619u;
            return h;
        }
    };

    VoxelKey keyOf(const std::array<float,3> &xyz) const;
    BucketId getOrCreate(const VoxelKey &key);

    float blockSize_;

    // key → sequential BucketId
    std::unordered_map<VoxelKey, BucketId, VoxelKeyHash> keyToId_;
    // BucketId → VoxelKey (reverse map for neighbour queries)
    std::vector<VoxelKey> idToKey_;
    // BucketId → set of Gaussian indices
    std::unordered_map<BucketId, std::unordered_set<int64_t>> buckets_;
    // BucketId → last active frame (-1 = inactive)
    std::unordered_map<BucketId, int32_t> activeSince_;

    static const std::unordered_set<int64_t> emptySet_;
};

} // namespace opensplat_live
