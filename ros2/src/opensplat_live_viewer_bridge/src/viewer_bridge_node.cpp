/// viewer_bridge_node.cpp
///
/// Subscribes to mapper and tracker topics and re-publishes them in a form
/// suitable for RViz and/or Rerun visualisation.
///
/// Topics consumed:
///   /tracking/odometry            → camera trajectory (Path + Marker)
///   /tracking/keyframe_candidate  → camera frustum marker array
///   /mapper/gaussian_stats        → text overlay / DiagnosticStatus
///   /mapper/map_update_stats      → timing log
///
/// Topics produced:
///   /viz/camera_path              (nav_msgs/Path)
///   /viz/frustum_markers          (visualization_msgs/MarkerArray)
///   /viz/map_bbox                 (visualization_msgs/MarkerArray)
///   /viz/gaussian_count_text      (visualization_msgs/Marker – text)

#include <deque>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "opensplat_live_interfaces/msg/keyframe.hpp"
#include "opensplat_live_interfaces/msg/gaussian_stats.hpp"
#include "opensplat_live_interfaces/msg/map_update_stats.hpp"

class ViewerBridgeNode : public rclcpp::Node {
public:
    ViewerBridgeNode()
        : Node("opensplat_live_viewer_bridge")
    {
        this->declare_parameter("max_trajectory_points", 2000);
        this->declare_parameter("frustum_scale", 0.15f);
        this->declare_parameter("map_frame", std::string("map"));
        maxTrajPts_  = get_parameter("max_trajectory_points").as_int();
        frustumScale_= get_parameter("frustum_scale").as_double();
        mapFrame_    = get_parameter("map_frame").as_string();

        auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

        subOdom_      = create_subscription<nav_msgs::msg::Odometry>(
            "/tracking/odometry", qos,
            std::bind(&ViewerBridgeNode::odomCallback, this, std::placeholders::_1));
        subKeyframe_  = create_subscription<opensplat_live_interfaces::msg::Keyframe>(
            "/tracking/keyframe_candidate", qos,
            std::bind(&ViewerBridgeNode::kfCallback, this, std::placeholders::_1));
        subGStats_    = create_subscription<opensplat_live_interfaces::msg::GaussianStats>(
            "/mapper/gaussian_stats", qos,
            std::bind(&ViewerBridgeNode::gStatsCallback, this, std::placeholders::_1));
        subUStats_    = create_subscription<opensplat_live_interfaces::msg::MapUpdateStats>(
            "/mapper/map_update_stats", qos,
            std::bind(&ViewerBridgeNode::uStatsCallback, this, std::placeholders::_1));

        pubPath_     = create_publisher<nav_msgs::msg::Path>("/viz/camera_path", qos);
        pubFrustums_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/viz/frustum_markers", qos);
        pubTextOverlay_ = create_publisher<visualization_msgs::msg::Marker>(
            "/viz/gaussian_count_text", qos);

        RCLCPP_INFO(get_logger(), "ViewerBridgeNode ready");
    }

private:
    int         maxTrajPts_;
    float       frustumScale_;
    std::string mapFrame_;

    nav_msgs::msg::Path              cameraPath_;
    visualization_msgs::msg::MarkerArray frustumMarkers_;
    uint32_t                         frustumSeq_ = 0;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom_;
    rclcpp::Subscription<opensplat_live_interfaces::msg::Keyframe>::SharedPtr subKeyframe_;
    rclcpp::Subscription<opensplat_live_interfaces::msg::GaussianStats>::SharedPtr subGStats_;
    rclcpp::Subscription<opensplat_live_interfaces::msg::MapUpdateStats>::SharedPtr subUStats_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubFrustums_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pubTextOverlay_;

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = msg->header;
        ps.header.frame_id = mapFrame_;
        ps.pose  = msg->pose.pose;

        cameraPath_.header = ps.header;
        cameraPath_.header.frame_id = mapFrame_;
        cameraPath_.poses.push_back(ps);

        if (static_cast<int>(cameraPath_.poses.size()) > maxTrajPts_)
            cameraPath_.poses.erase(cameraPath_.poses.begin());

        pubPath_->publish(cameraPath_);
    }

    void kfCallback(const opensplat_live_interfaces::msg::Keyframe::SharedPtr msg) {
        // Draw a simple LINE_LIST frustum for this keyframe
        visualization_msgs::msg::Marker m;
        m.header.stamp    = msg->header.stamp;
        m.header.frame_id = mapFrame_;
        m.ns     = "frustums";
        m.id     = static_cast<int>(frustumSeq_++);
        m.type   = visualization_msgs::msg::Marker::LINE_LIST;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = 0.01;
        m.color.r = 0.2f; m.color.g = 1.0f; m.color.b = 0.2f; m.color.a = 0.8f;
        m.lifetime = rclcpp::Duration(10, 0);

        // Camera origin
        const auto &p = msg->pose.pose.position;
        geometry_msgs::msg::Point origin;
        origin.x = p.x; origin.y = p.y; origin.z = p.z;

        // Four frustum corners at scale z=frustumScale_
        float s = frustumScale_;
        // Approximate: no rotation applied – just axis-aligned corners
        std::vector<geometry_msgs::msg::Point> corners(4);
        corners[0].x = p.x + s; corners[0].y = p.y + s*0.75f; corners[0].z = p.z + s;
        corners[1].x = p.x - s; corners[1].y = p.y + s*0.75f; corners[1].z = p.z + s;
        corners[2].x = p.x - s; corners[2].y = p.y - s*0.75f; corners[2].z = p.z + s;
        corners[3].x = p.x + s; corners[3].y = p.y - s*0.75f; corners[3].z = p.z + s;

        for (int i = 0; i < 4; ++i) {
            m.points.push_back(origin);
            m.points.push_back(corners[i]);
        }
        for (int i = 0; i < 4; ++i) {
            m.points.push_back(corners[i]);
            m.points.push_back(corners[(i+1)%4]);
        }

        frustumMarkers_.markers.push_back(m);
        pubFrustums_->publish(frustumMarkers_);
    }

    void gStatsCallback(
        const opensplat_live_interfaces::msg::GaussianStats::SharedPtr msg)
    {
        // Publish a text overlay showing current Gaussian count
        visualization_msgs::msg::Marker t;
        t.header.stamp    = msg->header.stamp;
        t.header.frame_id = mapFrame_;
        t.ns     = "stats_text";
        t.id     = 0;
        t.type   = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        t.action = visualization_msgs::msg::Marker::ADD;
        t.scale.z = 0.3;
        t.color.r = 1.0f; t.color.g = 1.0f; t.color.b = 1.0f; t.color.a = 1.0f;
        t.pose.orientation.w = 1.0;
        t.pose.position.z    = 0.5;
        t.text = "Gaussians: " + std::to_string(msg->total_gaussians)
               + " (stable=" + std::to_string(msg->stable_gaussians) + ")";
        pubTextOverlay_->publish(t);
    }

    void uStatsCallback(
        const opensplat_live_interfaces::msg::MapUpdateStats::SharedPtr msg)
    {
        RCLCPP_DEBUG(get_logger(),
            "KF %u update: %.1f ms (opt=%.1f spawn=%.1f)",
            msg->keyframe_id,
            msg->total_update_ms,
            msg->optimize_pass_ms,
            msg->spawn_pass_ms);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ViewerBridgeNode>());
    rclcpp::shutdown();
    return 0;
}
