#include "opensplat_live_mapper/keyframe_manager.hpp"
#include <cmath>

namespace opensplat_live {

KeyframeManager::KeyframeManager(const KeyframeConfig &cfg)
    : cfg_(cfg) {}

static float translationDist(float ax, float ay, float az,
                              float bx, float by, float bz) {
    float dx = ax-bx, dy = ay-by, dz = az-bz;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Quaternion angular distance in degrees
static float quatAngleDeg(float ax, float ay, float az, float aw,
                           float bx, float by, float bz, float bw) {
    float dot = ax*bx + ay*by + az*bz + aw*bw;
    dot = std::clamp(std::abs(dot), 0.0f, 1.0f);
    return 2.0f * std::acos(dot) * (180.0f / static_cast<float>(M_PI));
}

bool KeyframeManager::evaluate(const FrameInfo &info) {
    ++keyframeCount_; // total frames seen (not necessarily keyframes)

    // Force first frame to be a keyframe
    if (!hasLast_) {
        hasLast_ = true;
        last_ = info;
        lastKeyframeSec_ = info.timestampSec;
        return true;
    }

    bool becomeKf = false;

    // Translation threshold
    float dist = translationDist(info.tx, info.ty, info.tz,
                                 last_.tx, last_.ty, last_.tz);
    if (dist > cfg_.translationThresh) becomeKf = true;

    // Rotation threshold
    float angleDeg = quatAngleDeg(info.qx, info.qy, info.qz, info.qw,
                                  last_.qx, last_.qy, last_.qz, last_.qw);
    if (angleDeg > cfg_.rotationThreshDeg) becomeKf = true;

    // Novelty threshold
    if (info.noveltyRatio > cfg_.noveltyThresh) becomeKf = true;

    // Time interval
    double dt = info.timestampSec - lastKeyframeSec_;
    if (dt > cfg_.maxIntervalSec) becomeKf = true;

    // Confidence drop + recovery
    bool isLost = info.confidence < cfg_.minConfidence;
    if (wasLost_ && !isLost) becomeKf = true;
    wasLost_ = isLost;

    if (becomeKf) {
        last_ = info;
        lastKeyframeSec_ = info.timestampSec;
    }
    return becomeKf;
}

void KeyframeManager::reset() {
    hasLast_       = false;
    wasLost_       = false;
    keyframeCount_ = 0;
    lastKeyframeSec_ = -1.0;
}

} // namespace opensplat_live
