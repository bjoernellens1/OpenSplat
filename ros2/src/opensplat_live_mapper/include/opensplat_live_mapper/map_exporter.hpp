#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include "live_gaussian_map.hpp"

namespace opensplat_live {

struct ExporterConfig {
    std::string outputDir     = ".";
    std::string filenamePrefix= "map";
    bool        exportJson    = true;
    int         snapshotEveryN= 50; // keyframes between auto-snapshots
};

/// Asynchronously exports PLY snapshots and JSON metrics.
///
/// The export thread runs in the background and never blocks the mapper.
class MapExporter {
public:
    explicit MapExporter(const ExporterConfig &cfg = {});
    ~MapExporter();

    /// Request an async export snapshot.  Non-blocking.
    void requestSnapshot(const LiveGaussianMap &map, uint32_t keyframeId);

    /// Synchronously export to a specified path (blocks until done).
    void exportNow(const LiveGaussianMap &map,
                   uint32_t keyframeId,
                   const std::string &outputPath);

    /// Block until all queued exports have completed.
    void flush();

    const ExporterConfig& config() const { return cfg_; }

private:
    struct ExportRequest {
        // Snapshot of the parameter tensors (cloned to CPU for thread safety)
        torch::Tensor means;
        torch::Tensor scales;
        torch::Tensor quats;
        torch::Tensor featuresDc;
        torch::Tensor featuresRest;
        torch::Tensor opacities;
        uint32_t keyframeId;
        int64_t  totalGaussians;
        int64_t  stableCount;
    };

    void workerLoop();
    void writePly(const ExportRequest &req, const std::string &path);
    void writeJson(const ExportRequest &req, const std::string &path);

    ExporterConfig cfg_;

    std::thread            worker_;
    std::queue<ExportRequest> queue_;
    std::mutex             queueMutex_;
    std::condition_variable cv_;
    std::atomic<bool>      shutdown_{false};
};

} // namespace opensplat_live
