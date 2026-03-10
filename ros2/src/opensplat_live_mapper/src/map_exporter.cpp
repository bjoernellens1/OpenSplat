#include "opensplat_live_mapper/map_exporter.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

namespace opensplat_live {

// ─── PLY helper ──────────────────────────────────────────────────────────────
static void writePlyHeader(std::ostream &os, int64_t numPoints, int shRestBases) {
    os << "ply\nformat binary_little_endian 1.0\n"
       << "element vertex " << numPoints << "\n"
       << "property float x\nproperty float y\nproperty float z\n"
       << "property float nx\nproperty float ny\nproperty float nz\n"
       << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    for (int i = 0; i < shRestBases * 3; ++i)
        os << "property float f_rest_" << i << "\n";
    os << "property float opacity\n"
       << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
       << "property float rot_0\nproperty float rot_1\n"
       << "property float rot_2\nproperty float rot_3\n"
       << "end_header\n";
}

void MapExporter::writePly(const ExportRequest &req, const std::string &path) {
    std::ofstream os(path, std::ios::binary);
    if (!os.good()) {
        std::cerr << "[MapExporter] Cannot open " << path << " for writing\n";
        return;
    }

    int64_t N = req.means.size(0);
    if (N == 0) { std::cout << "[MapExporter] Empty map – skipping PLY\n"; return; }

    int shRestBases = static_cast<int>(req.featuresRest.size(1));
    writePlyHeader(os, N, shRestBases);

    auto meansAcc   = req.means.accessor<float,2>();
    auto scalesAcc  = req.scales.accessor<float,2>();
    auto quatsAcc   = req.quats.accessor<float,2>();
    auto dcAcc      = req.featuresDc.accessor<float,2>();
    auto opAcc      = req.opacities.accessor<float,2>();

    auto restReshaped = req.featuresRest.reshape({N, shRestBases * 3});
    auto restAcc = restReshaped.accessor<float,2>();

    for (int64_t i = 0; i < N; ++i) {
        auto write = [&](float v){ os.write(reinterpret_cast<const char*>(&v), 4); };
        // xyz
        write(meansAcc[i][0]); write(meansAcc[i][1]); write(meansAcc[i][2]);
        // nx ny nz (zero for now)
        write(0.f); write(0.f); write(0.f);
        // DC SH
        write(dcAcc[i][0]); write(dcAcc[i][1]); write(dcAcc[i][2]);
        // Rest SH
        for (int j = 0; j < shRestBases * 3; ++j) write(restAcc[i][j]);
        // Opacity
        write(opAcc[i][0]);
        // Scales
        write(scalesAcc[i][0]); write(scalesAcc[i][1]); write(scalesAcc[i][2]);
        // Quaternion (w,x,y,z)
        write(quatsAcc[i][0]); write(quatsAcc[i][1]);
        write(quatsAcc[i][2]); write(quatsAcc[i][3]);
    }
    std::cout << "[MapExporter] Saved " << N << " Gaussians to " << path << "\n";
}

void MapExporter::writeJson(const ExportRequest &req, const std::string &path) {
    std::ofstream os(path);
    if (!os.good()) return;
    os << "{\n"
       << "  \"keyframe_id\": " << req.keyframeId << ",\n"
       << "  \"total_gaussians\": " << req.totalGaussians << ",\n"
       << "  \"stable_gaussians\": " << req.stableCount << ",\n"
       << "  \"unstable_gaussians\": " << (req.totalGaussians - req.stableCount) << "\n"
       << "}\n";
}

// ─── Worker thread ────────────────────────────────────────────────────────────
MapExporter::MapExporter(const ExporterConfig &cfg) : cfg_(cfg) {
    worker_ = std::thread(&MapExporter::workerLoop, this);
}

MapExporter::~MapExporter() {
    {
        std::lock_guard<std::mutex> lk(queueMutex_);
        shutdown_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void MapExporter::workerLoop() {
    while (true) {
        ExportRequest req;
        {
            std::unique_lock<std::mutex> lk(queueMutex_);
            cv_.wait(lk, [this]{ return !queue_.empty() || shutdown_; });
            if (queue_.empty() && shutdown_) return;
            req = std::move(queue_.front());
            queue_.pop();
        }
        std::ostringstream base;
        base << cfg_.outputDir << "/" << cfg_.filenamePrefix
             << "_kf" << req.keyframeId;
        writePly(req, base.str() + ".ply");
        if (cfg_.exportJson) writeJson(req, base.str() + ".json");
    }
}

void MapExporter::requestSnapshot(const LiveGaussianMap &map, uint32_t keyframeId) {
    ExportRequest req;
    req.keyframeId      = keyframeId;
    req.totalGaussians  = map.size();
    req.stableCount     = map.stableMask().sum().item<int64_t>();
    // Clone tensors to CPU (thread-safe copy)
    req.means        = map.means().detach().cpu();
    req.scales       = map.scales().detach().cpu();
    req.quats        = map.quats().detach().cpu();
    req.featuresDc   = map.featuresDc().detach().cpu();
    req.featuresRest = map.featuresRest().detach().cpu();
    req.opacities    = map.opacities().detach().cpu();

    {
        std::lock_guard<std::mutex> lk(queueMutex_);
        queue_.push(std::move(req));
    }
    cv_.notify_one();
}

void MapExporter::exportNow(const LiveGaussianMap &map,
                             uint32_t keyframeId,
                             const std::string &outputPath) {
    ExportRequest req;
    req.keyframeId     = keyframeId;
    req.totalGaussians = map.size();
    req.stableCount    = map.stableMask().sum().item<int64_t>();
    req.means        = map.means().detach().cpu();
    req.scales       = map.scales().detach().cpu();
    req.quats        = map.quats().detach().cpu();
    req.featuresDc   = map.featuresDc().detach().cpu();
    req.featuresRest = map.featuresRest().detach().cpu();
    req.opacities    = map.opacities().detach().cpu();
    writePly(req, outputPath);
    if (cfg_.exportJson) {
        std::string jsonPath = outputPath;
        auto pos = jsonPath.rfind(".ply");
        if (pos != std::string::npos) jsonPath.replace(pos, 4, ".json");
        writeJson(req, jsonPath);
    }
}

void MapExporter::flush() {
    std::unique_lock<std::mutex> lk(queueMutex_);
    cv_.wait(lk, [this]{ return queue_.empty(); });
}

} // namespace opensplat_live
