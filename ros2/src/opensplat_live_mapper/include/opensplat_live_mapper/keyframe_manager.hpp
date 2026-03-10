#pragma once

#include <cstdint>
#include "live_gaussian_map.hpp"
#include "spatial_bucket_index.hpp"

namespace opensplat_live {

/// v1 defaults for indoor RGB-D live mapping
struct KeyframeConfig {
    float translationThresh = 0.10f;  // metres
    float rotationThreshDeg = 7.0f;   // degrees
    float noveltyThresh     = 0.25f;  // fraction of newly visible pixels
    float maxIntervalSec    = 0.5f;
    float minConfidence     = 0.3f;   // below this → force keyframe on recovery
};

/// Decides whether the current frame should become a keyframe.
class KeyframeManager {
public:
    explicit KeyframeManager(const KeyframeConfig &cfg = {});

    struct FrameInfo {
        double   timestampSec;
        float    tx, ty, tz;    // translation (metres)
        float    qx, qy, qz, qw;
        float    confidence;    // [0,1]
        float    noveltyRatio;  // [0,1]  fraction of new scene pixels
    };

    /// Returns true when a keyframe should be created for this frame.
    bool evaluate(const FrameInfo &info);

    uint32_t keyframeCount() const { return keyframeCount_; }

    void reset();

private:
    KeyframeConfig cfg_;
    FrameInfo      last_{};
    bool           hasLast_    = false;
    bool           wasLost_    = false;
    uint32_t       keyframeCount_ = 0;
    double         lastKeyframeSec_ = -1.0;
};

} // namespace opensplat_live
