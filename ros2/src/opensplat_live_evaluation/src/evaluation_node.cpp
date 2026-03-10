/// evaluation_node.cpp
///
/// Subscribes to tracking and mapper topics, logs trajectory and runtime
/// metrics, and computes ATE / RPE at shutdown time.
///
/// Trajectory log format (TUM compatible):
///   timestamp tx ty tz qx qy qz qw
///
/// At shutdown, an ATE RMSE estimate is printed to stdout and saved to CSV.
/// For full trajectory analysis, use the evo Python package:
///   evo_ape tum gt.txt estimated.txt -va --plot

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "opensplat_live_interfaces/msg/gaussian_stats.hpp"
#include "opensplat_live_interfaces/msg/map_update_stats.hpp"
#include "opensplat_live_interfaces/msg/tracking_status.hpp"

struct PoseRecord {
    double   t;
    double   tx, ty, tz;
    double   qx, qy, qz, qw;
};

struct MapRecord {
    double   t;
    uint32_t kf_id;
    uint32_t total;
    uint32_t stable;
    uint32_t unstable;
    uint32_t spawned;
    uint32_t pruned;
    float    update_ms;
};

class EvaluationNode : public rclcpp::Node {
public:
    EvaluationNode()
        : Node("opensplat_live_evaluation")
    {
        this->declare_parameter("output_dir", std::string("."));
        this->declare_parameter("gt_tum_file", std::string(""));
        outputDir_ = get_parameter("output_dir").as_string();
        gtFile_    = get_parameter("gt_tum_file").as_string();

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

        subOdom_   = create_subscription<nav_msgs::msg::Odometry>(
            "/tracking/odometry", qos,
            std::bind(&EvaluationNode::odomCallback, this, std::placeholders::_1));
        subGStats_ = create_subscription<opensplat_live_interfaces::msg::GaussianStats>(
            "/mapper/gaussian_stats", qos,
            std::bind(&EvaluationNode::gStatsCallback, this, std::placeholders::_1));
        subUStats_ = create_subscription<opensplat_live_interfaces::msg::MapUpdateStats>(
            "/mapper/map_update_stats", qos,
            std::bind(&EvaluationNode::uStatsCallback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "EvaluationNode ready – saving to %s", outputDir_.c_str());
    }

    ~EvaluationNode() {
        saveTrajectoryTum();
        saveMapMetricsCsv();
        if (!gtFile_.empty()) computeAte();
    }

private:
    std::string outputDir_;
    std::string gtFile_;

    std::vector<PoseRecord> poses_;
    std::vector<MapRecord>  mapRecords_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom_;
    rclcpp::Subscription<opensplat_live_interfaces::msg::GaussianStats>::SharedPtr subGStats_;
    rclcpp::Subscription<opensplat_live_interfaces::msg::MapUpdateStats>::SharedPtr subUStats_;

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        PoseRecord r;
        r.t  = rclcpp::Time(msg->header.stamp).seconds();
        r.tx = msg->pose.pose.position.x;
        r.ty = msg->pose.pose.position.y;
        r.tz = msg->pose.pose.position.z;
        r.qx = msg->pose.pose.orientation.x;
        r.qy = msg->pose.pose.orientation.y;
        r.qz = msg->pose.pose.orientation.z;
        r.qw = msg->pose.pose.orientation.w;
        poses_.push_back(r);
    }

    void gStatsCallback(
        const opensplat_live_interfaces::msg::GaussianStats::SharedPtr msg)
    {
        // Stored together with uStats in uStatsCallback; update last record
        if (!mapRecords_.empty()) {
            mapRecords_.back().total    = msg->total_gaussians;
            mapRecords_.back().stable   = msg->stable_gaussians;
            mapRecords_.back().unstable = msg->unstable_gaussians;
            mapRecords_.back().spawned  = msg->gaussians_spawned_last_kf;
            mapRecords_.back().pruned   = msg->gaussians_pruned_last_kf;
        }
    }

    void uStatsCallback(
        const opensplat_live_interfaces::msg::MapUpdateStats::SharedPtr msg)
    {
        MapRecord r{};
        r.t         = rclcpp::Time(msg->header.stamp).seconds();
        r.kf_id     = msg->keyframe_id;
        r.update_ms = msg->total_update_ms;
        mapRecords_.push_back(r);
    }

    void saveTrajectoryTum() {
        std::string path = outputDir_ + "/estimated_trajectory.txt";
        std::ofstream f(path);
        if (!f.good()) { RCLCPP_WARN(get_logger(), "Cannot write %s", path.c_str()); return; }
        f << std::fixed << std::setprecision(9);
        for (const auto &r : poses_)
            f << r.t << " " << r.tx << " " << r.ty << " " << r.tz << " "
              << r.qx << " " << r.qy << " " << r.qz << " " << r.qw << "\n";
        RCLCPP_INFO(get_logger(), "Saved %zu trajectory poses to %s",
                    poses_.size(), path.c_str());
    }

    void saveMapMetricsCsv() {
        std::string path = outputDir_ + "/map_metrics.csv";
        std::ofstream f(path);
        if (!f.good()) return;
        f << "timestamp,kf_id,total,stable,unstable,spawned,pruned,update_ms\n";
        for (const auto &r : mapRecords_)
            f << std::fixed << std::setprecision(6)
              << r.t << "," << r.kf_id << "," << r.total << "," << r.stable
              << "," << r.unstable << "," << r.spawned << "," << r.pruned
              << "," << r.update_ms << "\n";
        RCLCPP_INFO(get_logger(), "Saved %zu map records to %s",
                    mapRecords_.size(), path.c_str());
    }

    void computeAte() {
        // Load ground truth (TUM format: t tx ty tz qx qy qz qw)
        std::ifstream f(gtFile_);
        if (!f.good()) {
            RCLCPP_WARN(get_logger(), "GT file not found: %s", gtFile_.c_str());
            return;
        }
        struct GtPose { double t, tx, ty, tz; };
        std::vector<GtPose> gt;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            GtPose g; double qx,qy,qz,qw;
            ss >> g.t >> g.tx >> g.ty >> g.tz >> qx >> qy >> qz >> qw;
            gt.push_back(g);
        }
        if (gt.empty() || poses_.empty()) return;

        // Simple nearest-neighbour matching on timestamp
        double sumSq = 0;
        int    count = 0;
        for (const auto &est : poses_) {
            // Find nearest GT by time
            size_t best = 0;
            double bestDt = std::abs(est.t - gt[0].t);
            for (size_t i = 1; i < gt.size(); ++i) {
                double dt = std::abs(est.t - gt[i].t);
                if (dt < bestDt) { bestDt = dt; best = i; }
            }
            if (bestDt > 0.1) continue; // no match within 100 ms
            double dx = est.tx - gt[best].tx;
            double dy = est.ty - gt[best].ty;
            double dz = est.tz - gt[best].tz;
            sumSq += dx*dx + dy*dy + dz*dz;
            ++count;
        }
        if (count > 0) {
            double ate = std::sqrt(sumSq / count);
            RCLCPP_INFO(get_logger(),
                "ATE RMSE (translational): %.4f m over %d poses", ate, count);
            std::ofstream ate_f(outputDir_ + "/ate_rmse.txt");
            if (ate_f.good()) ate_f << ate << "\n";
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EvaluationNode>());
    rclcpp::shutdown();
    return 0;
}
