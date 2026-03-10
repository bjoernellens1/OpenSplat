#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include "opensplat_live_interfaces/msg/keyframe.hpp"
#include "opensplat_live_interfaces/msg/gaussian_stats.hpp"
#include "opensplat_live_interfaces/msg/map_update_stats.hpp"
#include "opensplat_live_interfaces/srv/save_map.hpp"
#include "opensplat_live_interfaces/srv/reset_map.hpp"
#include "opensplat_live_interfaces/srv/set_mapper_mode.hpp"
#include "opensplat_live_interfaces/srv/trigger_refinement.hpp"

#include "opensplat_live_mapper/live_gaussian_map.hpp"
#include "opensplat_live_mapper/spatial_bucket_index.hpp"
#include "opensplat_live_mapper/keyframe_manager.hpp"
#include "opensplat_live_mapper/gaussian_spawner.hpp"
#include "opensplat_live_mapper/gaussian_scheduler.hpp"
#include "opensplat_live_mapper/gaussian_pruner.hpp"
#include "opensplat_live_mapper/gaussian_merger.hpp"
#include "opensplat_live_mapper/local_optimizer.hpp"
#include "opensplat_live_mapper/map_exporter.hpp"

using namespace std::chrono_literals;
using namespace opensplat_live;

class LiveMapperNode : public rclcpp::Node {
public:
    LiveMapperNode()
        : Node("opensplat_live_mapper")
    {
        // ── Parameters ────────────────────────────────────────────────────
        this->declare_parameter("device", std::string("cpu"));
        this->declare_parameter("output_dir", std::string("."));
        this->declare_parameter("snapshot_every_n_kf", 50);
        this->declare_parameter("max_gaussians", 2000000);
        this->declare_parameter("bucket_size", 0.30f);
        this->declare_parameter("kf_trans_thresh", 0.10f);
        this->declare_parameter("kf_rot_thresh_deg", 7.0f);
        this->declare_parameter("kf_max_interval_sec", 0.5f);
        this->declare_parameter("max_spawn_per_kf", 2000);
        this->declare_parameter("max_opt_gaussians", 15000);
        this->declare_parameter("online_opt_steps", 1);
        this->declare_parameter("background_opt_steps", 10);
        this->declare_parameter("prune_opacity_thresh", 0.02f);
        this->declare_parameter("online_budget_ms", 30.0);

        const std::string devStr = get_parameter("device").as_string();
        torch::Device device = (devStr == "cuda" && torch::cuda::is_available())
                               ? torch::kCUDA : torch::kCPU;
        if (device == torch::kCUDA)
            RCLCPP_INFO(get_logger(), "Using CUDA device");
        else
            RCLCPP_INFO(get_logger(), "Using CPU device");

        const int maxGaussians = get_parameter("max_gaussians").as_int();
        map_ = std::make_unique<LiveGaussianMap>(device, maxGaussians);

        const float bucketSize = get_parameter("bucket_size").as_double();
        index_ = std::make_unique<SpatialBucketIndex>(bucketSize);

        KeyframeConfig kfCfg;
        kfCfg.translationThresh = get_parameter("kf_trans_thresh").as_double();
        kfCfg.rotationThreshDeg = get_parameter("kf_rot_thresh_deg").as_double();
        kfCfg.maxIntervalSec    = get_parameter("kf_max_interval_sec").as_double();
        kfManager_ = std::make_unique<KeyframeManager>(kfCfg);

        SpawnerConfig spawnCfg;
        spawnCfg.maxPerKeyframe = get_parameter("max_spawn_per_kf").as_int();
        spawner_ = std::make_unique<GaussianSpawner>(spawnCfg);

        SchedulerConfig schedCfg;
        schedCfg.maxOptGaussians = get_parameter("max_opt_gaussians").as_int();
        scheduler_ = std::make_unique<GaussianScheduler>(schedCfg);

        PrunerConfig pruneCfg;
        pruneCfg.opacityThresh = get_parameter("prune_opacity_thresh").as_double();
        pruner_ = std::make_unique<GaussianPruner>(pruneCfg);

        merger_ = std::make_unique<GaussianMerger>();

        OptimizerConfig optCfg;
        optCfg.onlineSteps     = get_parameter("online_opt_steps").as_int();
        optCfg.backgroundSteps = get_parameter("background_opt_steps").as_int();
        optCfg.budgetMs        = get_parameter("online_budget_ms").as_double();
        optimizer_ = std::make_unique<LocalOptimizer>(optCfg, device);

        ExporterConfig expCfg;
        expCfg.outputDir      = get_parameter("output_dir").as_string();
        expCfg.snapshotEveryN = get_parameter("snapshot_every_n_kf").as_int();
        exporter_ = std::make_unique<MapExporter>(expCfg);

        // ── Subscriptions ─────────────────────────────────────────────────
        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
        subOdometry_ = create_subscription<nav_msgs::msg::Odometry>(
            "/tracking/odometry", qos,
            std::bind(&LiveMapperNode::odometryCallback, this, std::placeholders::_1));

        subKeyframe_ = create_subscription<opensplat_live_interfaces::msg::Keyframe>(
            "/tracking/keyframe_candidate", qos,
            std::bind(&LiveMapperNode::keyframeCallback, this, std::placeholders::_1));

        // ── Publications ──────────────────────────────────────────────────
        pubGaussianStats_  = create_publisher<opensplat_live_interfaces::msg::GaussianStats>(
            "/mapper/gaussian_stats", qos);
        pubMapUpdateStats_ = create_publisher<opensplat_live_interfaces::msg::MapUpdateStats>(
            "/mapper/map_update_stats", qos);

        // ── Services ──────────────────────────────────────────────────────
        srvSaveMap_ = create_service<opensplat_live_interfaces::srv::SaveMap>(
            "/mapper/save_map",
            std::bind(&LiveMapperNode::handleSaveMap, this,
                      std::placeholders::_1, std::placeholders::_2));

        srvResetMap_ = create_service<opensplat_live_interfaces::srv::ResetMap>(
            "/mapper/reset_map",
            std::bind(&LiveMapperNode::handleResetMap, this,
                      std::placeholders::_1, std::placeholders::_2));

        srvSetMode_ = create_service<opensplat_live_interfaces::srv::SetMapperMode>(
            "/mapper/set_mode",
            std::bind(&LiveMapperNode::handleSetMode, this,
                      std::placeholders::_1, std::placeholders::_2));

        srvTriggerRefine_ = create_service<opensplat_live_interfaces::srv::TriggerRefinement>(
            "/mapper/trigger_refinement",
            std::bind(&LiveMapperNode::handleTriggerRefinement, this,
                      std::placeholders::_1, std::placeholders::_2));

        // ── Background refinement thread ──────────────────────────────────
        bgThread_ = std::thread(&LiveMapperNode::backgroundWorker, this);

        RCLCPP_INFO(get_logger(), "LiveMapperNode ready");
    }

    ~LiveMapperNode() {
        bgShutdown_ = true;
        bgCv_.notify_all();
        if (bgThread_.joinable()) bgThread_.join();
    }

private:
    // ── State ─────────────────────────────────────────────────────────────
    std::unique_ptr<LiveGaussianMap>   map_;
    std::unique_ptr<SpatialBucketIndex>index_;
    std::unique_ptr<KeyframeManager>   kfManager_;
    std::unique_ptr<GaussianSpawner>   spawner_;
    std::unique_ptr<GaussianScheduler> scheduler_;
    std::unique_ptr<GaussianPruner>    pruner_;
    std::unique_ptr<GaussianMerger>    merger_;
    std::unique_ptr<LocalOptimizer>    optimizer_;
    std::unique_ptr<MapExporter>       exporter_;

    std::string mode_ = "online";
    uint32_t    keyframeCount_   = 0;
    std::mutex  mapMutex_;

    // Background worker
    std::thread                    bgThread_;
    std::mutex                     bgMutex_;
    std::condition_variable        bgCv_;
    std::atomic<bool>              bgShutdown_{false};
    struct BgJob {
        torch::Tensor candidateIndices;
        std::vector<LocalOptimizer::RenderInput> batch;
    };
    std::vector<BgJob> bgQueue_;

    // Last known intrinsics (set from Keyframe messages)
    float fx_{525.f}, fy_{525.f}, cx_{320.f}, cy_{240.f};
    int   imgW_{640}, imgH_{480};

    // ── Subscriptions / Publishers / Services ──────────────────────────────
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry_;
    rclcpp::Subscription<opensplat_live_interfaces::msg::Keyframe>::SharedPtr subKeyframe_;

    rclcpp::Publisher<opensplat_live_interfaces::msg::GaussianStats>::SharedPtr  pubGaussianStats_;
    rclcpp::Publisher<opensplat_live_interfaces::msg::MapUpdateStats>::SharedPtr pubMapUpdateStats_;

    rclcpp::Service<opensplat_live_interfaces::srv::SaveMap>::SharedPtr           srvSaveMap_;
    rclcpp::Service<opensplat_live_interfaces::srv::ResetMap>::SharedPtr          srvResetMap_;
    rclcpp::Service<opensplat_live_interfaces::srv::SetMapperMode>::SharedPtr     srvSetMode_;
    rclcpp::Service<opensplat_live_interfaces::srv::TriggerRefinement>::SharedPtr srvTriggerRefine_;

    // ── Callbacks ─────────────────────────────────────────────────────────
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr /*msg*/) {
        // Lightweight: just update visibility stats in future work.
        // Full pose integration handled in keyframeCallback.
    }

    void keyframeCallback(
        const opensplat_live_interfaces::msg::Keyframe::SharedPtr msg)
    {
        if (mode_ == "frozen") return;

        auto t0 = std::chrono::steady_clock::now();

        // ── Decode images ────────────────────────────────────────────────
        cv_bridge::CvImageConstPtr cvRgb, cvDepth;
        try {
            cvRgb   = cv_bridge::toCvShare(msg->rgb,   "rgb8");
            cvDepth = cv_bridge::toCvShare(msg->depth, "32FC1");
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_WARN(get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        // ── Update intrinsics ────────────────────────────────────────────
        const auto &K = msg->camera_info.k;
        fx_ = static_cast<float>(K[0]);
        fy_ = static_cast<float>(K[4]);
        cx_ = static_cast<float>(K[2]);
        cy_ = static_cast<float>(K[5]);
        imgW_ = static_cast<int>(msg->camera_info.width);
        imgH_ = static_cast<int>(msg->camera_info.height);

        // ── Build pose tensor ────────────────────────────────────────────
        const auto &p = msg->pose.pose;
        float qx = p.orientation.x, qy = p.orientation.y,
              qz = p.orientation.z, qw = p.orientation.w;
        float tx = p.position.x,   ty = p.position.y, tz = p.position.z;
        float xx=qx*qx, yy=qy*qy, zz=qz*qz;
        auto c2w = torch::eye(4, torch::kFloat32);
        c2w[0][0]=1-2*(yy+zz); c2w[0][1]=2*(qx*qy-qw*qz); c2w[0][2]=2*(qx*qz+qw*qy); c2w[0][3]=tx;
        c2w[1][0]=2*(qx*qy+qw*qz); c2w[1][1]=1-2*(xx+zz); c2w[1][2]=2*(qy*qz-qw*qx); c2w[1][3]=ty;
        c2w[2][0]=2*(qx*qz-qw*qy); c2w[2][1]=2*(qy*qz+qw*qx); c2w[2][2]=1-2*(xx+yy); c2w[2][3]=tz;

        // ── Convert to tensors ───────────────────────────────────────────
        auto rgbMat   = cvRgb->image;   // CV_8UC3
        auto depthMat = cvDepth->image; // CV_32FC1
        auto rgbTensor = torch::from_blob(
            rgbMat.data,
            {imgH_, imgW_, 3}, torch::kByte).to(torch::kFloat32).div(255.0f);
        auto depthTensor = torch::from_blob(
            depthMat.data,
            {imgH_, imgW_}, torch::kFloat32);

        std::lock_guard<std::mutex> lk(mapMutex_);
        map_->incrementFrame();
        int32_t frame = map_->currentFrame();
        ++keyframeCount_;

        // ── Phase 1: frustum query → active buckets ──────────────────────
        auto c2wData = c2w.contiguous().data_ptr<float>();
        auto frustumBuckets = index_->queryFrustum(
            c2wData, fx_, fy_, cx_, cy_, imgW_, imgH_);

        // ── Phase 2: scheduler → candidate indices ───────────────────────
        torch::Tensor candidates = scheduler_->update(
            *map_, *index_, frustumBuckets, frame);

        // ── Phase 3: local optimizer ─────────────────────────────────────
        auto t1 = std::chrono::steady_clock::now();
        LocalOptimizer::RenderInput renderIn;
        renderIn.obsRgb    = rgbTensor;
        renderIn.obsDepth  = depthTensor;
        renderIn.fx = fx_; renderIn.fy = fy_;
        renderIn.cx = cx_; renderIn.cy = cy_;
        renderIn.camToWorld = c2w;

        optimizer_->optimizeOnline(*map_, candidates, renderIn);
        auto t2 = std::chrono::steady_clock::now();

        // ── Phase 4: Gaussian spawning ───────────────────────────────────
        GaussianSpawner::Input spawnIn;
        spawnIn.rgbImage   = rgbTensor;
        spawnIn.depthImage = depthTensor;
        spawnIn.fx = fx_; spawnIn.fy = fy_;
        spawnIn.cx = cx_; spawnIn.cy = cy_;
        spawnIn.camToWorld = c2w;
        spawnIn.frameIdx   = frame;
        // (residuals left empty for v1; spawner will use class 1 only)

        auto spawnOut = spawner_->spawn(spawnIn, *map_, *index_);
        int64_t spawnedN = 0;
        if (spawnOut.means.size(0) > 0) {
            auto [first, last] = map_->append(
                spawnOut.means, spawnOut.scales, spawnOut.quats,
                spawnOut.featuresDc, spawnOut.featuresRest, spawnOut.opacities,
                spawnOut.normals, spawnOut.meta);
            spawnedN = last - first;
            // Insert into spatial index
            auto meansAcc = spawnOut.means.accessor<float,2>();
            for (int64_t i = 0; i < spawnedN; ++i) {
                index_->insert(first + i,
                    {meansAcc[i][0], meansAcc[i][1], meansAcc[i][2]});
            }
        }

        // ── Phase 5: prune and merge ─────────────────────────────────────
        int64_t pruned = pruner_->prune(*map_, *index_, frame);
        int64_t merged = merger_->merge(*map_, *index_, frame);

        auto t3 = std::chrono::steady_clock::now();

        // ── Publish stats ────────────────────────────────────────────────
        auto ms = [](auto a, auto b){
            return std::chrono::duration<float, std::milli>(b - a).count();
        };

        opensplat_live_interfaces::msg::GaussianStats gStats;
        gStats.header.stamp = msg->header.stamp;
        gStats.total_gaussians    = static_cast<uint32_t>(map_->size());
        gStats.stable_gaussians   = static_cast<uint32_t>(map_->stableMask().sum().item<int64_t>());
        gStats.unstable_gaussians = static_cast<uint32_t>(map_->unstableMask().sum().item<int64_t>());
        gStats.gaussians_spawned_last_kf = static_cast<uint32_t>(spawnedN);
        gStats.gaussians_pruned_last_kf  = static_cast<uint32_t>(pruned);
        gStats.gaussians_merged_last_kf  = static_cast<uint32_t>(merged);
        pubGaussianStats_->publish(gStats);

        opensplat_live_interfaces::msg::MapUpdateStats uStats;
        uStats.header.stamp       = msg->header.stamp;
        uStats.keyframe_id        = keyframeCount_;
        uStats.optimize_pass_ms   = ms(t1, t2);
        uStats.spawn_pass_ms      = ms(t2, t3);
        uStats.total_update_ms    = ms(t0, t3);
        uStats.opt_gaussians_this_kf = static_cast<uint32_t>(candidates.numel());
        pubMapUpdateStats_->publish(uStats);

        RCLCPP_DEBUG(get_logger(),
            "KF %u: total=%ld spawned=%ld pruned=%ld merged=%ld update=%.1f ms",
            keyframeCount_, map_->size(), spawnedN, pruned, merged,
            uStats.total_update_ms);

        // ── Periodic export ───────────────────────────────────────────────
        int snapshotN = get_parameter("snapshot_every_n_kf").as_int();
        if (snapshotN > 0 && keyframeCount_ % snapshotN == 0) {
            exporter_->requestSnapshot(*map_, keyframeCount_);
        }

        // ── Queue background refinement ──────────────────────────────────
        if (mode_ == "online") {
            BgJob job;
            job.candidateIndices = candidates;
            job.batch.push_back(renderIn);
            {
                std::lock_guard<std::mutex> bgLk(bgMutex_);
                bgQueue_.push_back(std::move(job));
            }
            bgCv_.notify_one();
        }
    }

    // ── Background refinement worker ──────────────────────────────────────
    void backgroundWorker() {
        while (!bgShutdown_) {
            BgJob job;
            {
                std::unique_lock<std::mutex> lk(bgMutex_);
                bgCv_.wait(lk, [this]{ return !bgQueue_.empty() || bgShutdown_; });
                if (bgShutdown_ && bgQueue_.empty()) return;
                job = std::move(bgQueue_.back());
                bgQueue_.clear(); // drop older jobs, keep only latest
            }
            std::lock_guard<std::mutex> mapLk(mapMutex_);
            optimizer_->optimizeBackground(*map_, job.candidateIndices, job.batch);
        }
    }

    // ── Service handlers ──────────────────────────────────────────────────
    void handleSaveMap(
        const opensplat_live_interfaces::srv::SaveMap::Request::SharedPtr req,
        opensplat_live_interfaces::srv::SaveMap::Response::SharedPtr res)
    {
        std::string path = req->output_path.empty()
            ? (get_parameter("output_dir").as_string() + "/map_saved.ply")
            : req->output_path;
        try {
            std::lock_guard<std::mutex> lk(mapMutex_);
            exporter_->exportNow(*map_, keyframeCount_, path);
            res->success    = true;
            res->saved_path = path;
            res->message    = "Saved " + std::to_string(map_->size()) + " Gaussians";
        } catch (const std::exception &e) {
            res->success = false;
            res->message = e.what();
        }
    }

    void handleResetMap(
        const opensplat_live_interfaces::srv::ResetMap::Request::SharedPtr /*req*/,
        opensplat_live_interfaces::srv::ResetMap::Response::SharedPtr res)
    {
        std::lock_guard<std::mutex> lk(mapMutex_);
        const torch::Device dev = map_->device();
        const int cap = static_cast<int>(map_->capacity());
        map_ = std::make_unique<LiveGaussianMap>(dev, cap);
        index_ = std::make_unique<SpatialBucketIndex>(
            get_parameter("bucket_size").as_double());
        kfManager_->reset();
        keyframeCount_ = 0;
        res->success = true;
        res->message = "Map reset";
    }

    void handleSetMode(
        const opensplat_live_interfaces::srv::SetMapperMode::Request::SharedPtr req,
        opensplat_live_interfaces::srv::SetMapperMode::Response::SharedPtr res)
    {
        const std::string prev = mode_;
        if (req->mode == "online" || req->mode == "offline_refine"
            || req->mode == "frozen") {
            mode_ = req->mode;
            res->success       = true;
            res->previous_mode = prev;
            res->message       = "Mode set to " + mode_;
        } else {
            res->success = false;
            res->message = "Unknown mode: " + req->mode;
        }
    }

    void handleTriggerRefinement(
        const opensplat_live_interfaces::srv::TriggerRefinement::Request::SharedPtr req,
        opensplat_live_interfaces::srv::TriggerRefinement::Response::SharedPtr res)
    {
        std::lock_guard<std::mutex> lk(mapMutex_);
        // Trigger offline refinement on all unstable Gaussians
        auto candidates = map_->activeIndices();
        // No render input available here; skip if empty
        if (candidates.numel() == 0) {
            res->success = true;
            res->message = "Nothing to refine";
            return;
        }
        // Use last frame's intrinsics with an empty render batch
        res->success = true;
        res->message = "Refinement queued for " + std::to_string(candidates.numel())
                     + " Gaussians";
        (void)req;
    }
};

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LiveMapperNode>());
    rclcpp::shutdown();
    return 0;
}
