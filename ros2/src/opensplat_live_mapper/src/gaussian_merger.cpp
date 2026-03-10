#include "opensplat_live_mapper/gaussian_merger.hpp"
#include <cmath>

namespace opensplat_live {

GaussianMerger::GaussianMerger(const MergerConfig &cfg) : cfg_(cfg) {}

static float cosAngle(const float *a, const float *b) {
    float dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    float la  = std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
    float lb  = std::sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
    if (la < 1e-6f || lb < 1e-6f) return 0.0f;
    return std::clamp(dot / (la * lb), -1.0f, 1.0f);
}

int64_t GaussianMerger::merge(LiveGaussianMap &map,
                               SpatialBucketIndex &index,
                               int32_t /*currentFrame*/) {
    int64_t N = map.size();
    if (N < 2) return 0;

    auto meansCpu     = map.means().detach().cpu();       // [N,3]
    auto scalesCpu    = map.scales().detach().cpu();      // [N,3] log-space
    auto normalsCpu   = map.normals().detach().cpu();     // [N,3]
    auto featuresDcCpu= map.featuresDc().detach().cpu();  // [N,3]

    auto meansAcc  = meansCpu.accessor<float,2>();
    auto scalesAcc = scalesCpu.accessor<float,2>();
    auto normsAcc  = normalsCpu.accessor<float,2>();
    auto colAcc    = featuresDcCpu.accessor<float,2>();

    const float angleThreshCos = std::cos(cfg_.normalAngleThreshDeg * 3.14159265f / 180.0f);

    torch::Tensor deletedMask = torch::zeros({N}, torch::kBool);
    auto delAcc = deletedMask.accessor<bool,1>();

    int mergeCount = 0;

    // Iterate over populated buckets
    for (BucketId bid : index.activeBuckets()) {
        if (mergeCount >= cfg_.maxMergesPerKf) break;
        const auto &ids = index.indicesInBucket(bid);
        if (ids.size() < 2) continue;

        std::vector<int64_t> idVec(ids.begin(), ids.end());
        for (size_t a = 0; a < idVec.size() && mergeCount < cfg_.maxMergesPerKf; ++a) {
            int64_t ia = idVec[a];
            if (delAcc[ia]) continue;

            for (size_t b = a+1; b < idVec.size(); ++b) {
                int64_t ib = idVec[b];
                if (delAcc[ib]) continue;

                // Distance check
                float dx = meansAcc[ia][0]-meansAcc[ib][0];
                float dy = meansAcc[ia][1]-meansAcc[ib][1];
                float dz = meansAcc[ia][2]-meansAcc[ib][2];
                float dist = std::sqrt(dx*dx+dy*dy+dz*dz);

                // Local scale: mean of exp(scales)
                float sa = (std::exp(scalesAcc[ia][0])+std::exp(scalesAcc[ia][1])
                            +std::exp(scalesAcc[ia][2])) / 3.0f;
                float sb = (std::exp(scalesAcc[ib][0])+std::exp(scalesAcc[ib][1])
                            +std::exp(scalesAcc[ib][2])) / 3.0f;
                float scaleRef = (sa + sb) * 0.5f * cfg_.distanceThreshFactor;
                if (dist > scaleRef) continue;

                // Normal angle check
                float na[3]={normsAcc[ia][0],normsAcc[ia][1],normsAcc[ia][2]};
                float nb[3]={normsAcc[ib][0],normsAcc[ib][1],normsAcc[ib][2]};
                if (cosAngle(na, nb) < angleThreshCos) continue;

                // Colour distance
                float cr = colAcc[ia][0]-colAcc[ib][0];
                float cg = colAcc[ia][1]-colAcc[ib][1];
                float cb = colAcc[ia][2]-colAcc[ib][2];
                if (std::sqrt(cr*cr+cg*cg+cb*cb) > cfg_.colorDistThresh) continue;

                // Merge b into a (average means, keep a's rest)
                // (Full parameter averaging would require grad surgery;
                //  for v1 we simply remove the weaker one.)
                delAcc[ib] = true;
                ++mergeCount;
            }
        }
    }

    if (mergeCount == 0) return 0;

    // Update spatial index
    auto meansCpuFinal = meansCpu;
    auto meansAccF     = meansCpuFinal.accessor<float,2>();
    for (int64_t i = 0; i < N; ++i) {
        if (delAcc[i]) {
            std::array<float,3> xyz{meansAccF[i][0], meansAccF[i][1], meansAccF[i][2]};
            index.remove(i, xyz);
        }
    }
    map.remove(deletedMask);
    return mergeCount;
}

} // namespace opensplat_live
