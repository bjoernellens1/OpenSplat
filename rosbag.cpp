#ifdef HAVE_ROSBAG2

#include "rosbag.hpp"
#include "cv_utils.hpp"

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/serialization.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace rb {

namespace {

// --------------------------------------------------------------------------
// Convert a sensor_msgs/Image to an RGB cv::Mat.
// --------------------------------------------------------------------------
cv::Mat rosImageToMat(const sensor_msgs::msg::Image &msg) {
    const std::string &enc = msg.encoding;

    // Determine OpenCV type
    int cv_type = CV_8UC3;
    if (enc == "mono8")                                cv_type = CV_8UC1;
    else if (enc == "bgr8"  || enc == "rgb8")          cv_type = CV_8UC3;
    else if (enc == "bgra8" || enc == "rgba8")         cv_type = CV_8UC4;
    else if (enc == "16UC1")                           cv_type = CV_16UC1;
    else
        throw std::runtime_error("Unsupported image encoding: " + enc);

    // Wrap the message buffer (zero-copy)
    cv::Mat mat(static_cast<int>(msg.height),
                static_cast<int>(msg.width),
                cv_type,
                const_cast<uint8_t *>(msg.data.data()),
                msg.step);

    cv::Mat rgb;
    if      (enc == "bgr8")   cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else if (enc == "bgra8")  cv::cvtColor(mat, rgb, cv::COLOR_BGRA2RGB);
    else if (enc == "rgba8")  cv::cvtColor(mat, rgb, cv::COLOR_RGBA2RGB);
    else if (enc == "mono8")  cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    else                      mat.copyTo(rgb);   // rgb8 / 16UC1

    return rgb;
}

// --------------------------------------------------------------------------
// Build a 4×4 cam-to-world matrix from a quaternion + translation.
// Follows ROS REP-103 (x-forward, y-left, z-up body frame).
// --------------------------------------------------------------------------
torch::Tensor poseToMatrix(float tx, float ty, float tz,
                            float qx, float qy, float qz, float qw) {
    float xx = qx*qx, yy = qy*qy, zz = qz*qz;
    float xy = qx*qy, xz = qx*qz, yz = qy*qz;
    float wx = qw*qx, wy = qw*qy, wz = qw*qz;

    torch::Tensor m = torch::eye(4, torch::kFloat32);
    m[0][0] = 1.0f - 2.0f*(yy + zz);
    m[0][1] = 2.0f*(xy - wz);
    m[0][2] = 2.0f*(xz + wy);
    m[0][3] = tx;
    m[1][0] = 2.0f*(xy + wz);
    m[1][1] = 1.0f - 2.0f*(xx + zz);
    m[1][2] = 2.0f*(yz - wx);
    m[1][3] = ty;
    m[2][0] = 2.0f*(xz - wy);
    m[2][1] = 2.0f*(yz + wx);
    m[2][2] = 1.0f - 2.0f*(xx + yy);
    m[2][3] = tz;
    return m;
}

// --------------------------------------------------------------------------
// Simple timestamped pose record
// --------------------------------------------------------------------------
struct StampedPose {
    uint64_t stamp_ns;
    float tx, ty, tz;
    float qx, qy, qz, qw;
};

// --------------------------------------------------------------------------
// Find the pose whose timestamp is nearest to stamp_ns.
// Assumes poses is sorted by stamp_ns.
// --------------------------------------------------------------------------
const StampedPose *nearestPose(const std::vector<StampedPose> &poses,
                                uint64_t stamp_ns) {
    if (poses.empty()) return nullptr;
    auto it = std::lower_bound(
        poses.begin(), poses.end(), stamp_ns,
        [](const StampedPose &p, uint64_t t) { return p.stamp_ns < t; });
    if (it == poses.end())   return &poses.back();
    if (it == poses.begin()) return &poses.front();
    auto prev = std::prev(it);
    return (stamp_ns - prev->stamp_ns < it->stamp_ns - stamp_ns)
               ? &(*prev)
               : &(*it);
}

} // anonymous namespace

// ============================================================================
// Main entry point
// ============================================================================
InputData inputDataFromRosBag(const std::string &bagPath,
                               const std::string &imageTopic,
                               const std::string &cameraInfoTopic,
                               const std::string &poseTopic) {
    if (!fs::exists(bagPath))
        throw std::runtime_error("ROS2 bag path does not exist: " + bagPath);

    // ── Determine storage format (sqlite3 or mcap) ────────────────────────
    std::string storage_id = "sqlite3";
    if (fs::is_directory(bagPath)) {
        for (const auto &entry : fs::directory_iterator(bagPath)) {
            if (entry.path().extension() == ".mcap") {
                storage_id = "mcap";
                break;
            }
        }
    } else if (fs::path(bagPath).extension() == ".mcap") {
        storage_id = "mcap";
    }

    rosbag2_storage::StorageOptions storage_opts;
    storage_opts.uri        = bagPath;
    storage_opts.storage_id = storage_id;

    // ── First pass: discover available topics ─────────────────────────────
    std::string img_topic        = imageTopic;
    std::string info_topic       = cameraInfoTopic;
    std::string pose_topic_used  = poseTopic;
    bool        is_compressed    = false;
    bool        has_pose_topic   = false;

    {
        rosbag2_cpp::Reader reader;
        reader.open(storage_opts);
        auto topics = reader.get_all_topics_and_types();

        // Auto-detect image topic
        if (img_topic.empty()) {
            for (const auto &t : topics) {
                if (t.type == "sensor_msgs/msg/Image" ||
                    t.type == "sensor_msgs/msg/CompressedImage") {
                    img_topic = t.name;
                    std::cout << "[rosbag] Auto-detected image topic: "
                              << img_topic << " [" << t.type << "]" << std::endl;
                    break;
                }
            }
        }
        if (img_topic.empty()) {
            std::string available;
            for (const auto &t : topics)
                available += "  " + t.name + " [" + t.type + "]\n";
            throw std::runtime_error(
                "No image topic found in bag. Available topics:\n" + available);
        }

        // Check whether the image topic carries compressed images
        for (const auto &t : topics) {
            if (t.name == img_topic &&
                t.type == "sensor_msgs/msg/CompressedImage") {
                is_compressed = true;
                break;
            }
        }

        // Auto-detect camera_info topic
        if (info_topic.empty()) {
            // First try: sibling "camera_info" of the image topic
            fs::path sibling = fs::path(img_topic).parent_path() / "camera_info";
            for (const auto &t : topics) {
                if (t.name == sibling.string() &&
                    t.type == "sensor_msgs/msg/CameraInfo") {
                    info_topic = t.name;
                    break;
                }
            }
            // Fallback: any CameraInfo topic
            if (info_topic.empty()) {
                for (const auto &t : topics) {
                    if (t.type == "sensor_msgs/msg/CameraInfo") {
                        info_topic = t.name;
                        std::cout << "[rosbag] Auto-detected camera_info topic: "
                                  << info_topic << std::endl;
                        break;
                    }
                }
            }
        }
        if (info_topic.empty()) {
            throw std::runtime_error(
                "No sensor_msgs/msg/CameraInfo topic found in bag. "
                "Please specify --camera-info-topic.");
        }

        // Auto-detect pose topic
        if (pose_topic_used.empty()) {
            for (const auto &t : topics) {
                if (t.type == "geometry_msgs/msg/PoseStamped") {
                    pose_topic_used = t.name;
                    std::cout << "[rosbag] Auto-detected pose topic: "
                              << pose_topic_used << std::endl;
                    break;
                }
            }
        }
        has_pose_topic = !pose_topic_used.empty();
        if (!has_pose_topic) {
            std::cerr << "[rosbag] Warning: no geometry_msgs/msg/PoseStamped "
                         "topic found. Identity camera poses will be used.\n"
                         "         Consider running SfM (e.g., COLMAP) on the "
                         "extracted frames to obtain accurate poses.\n";
        }
    }

    // ── Second pass: collect CameraInfo (first occurrence) ────────────────
    sensor_msgs::msg::CameraInfo cam_info;
    bool got_cam_info = false;
    {
        rosbag2_cpp::Reader reader;
        reader.open(storage_opts);
        rclcpp::Serialization<sensor_msgs::msg::CameraInfo> ser;
        while (reader.has_next() && !got_cam_info) {
            auto msg = reader.read_next();
            if (msg->topic_name == info_topic) {
                rclcpp::SerializedMessage sm(*msg->serialized_data);
                auto info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
                ser.deserialize_message(&sm, info_msg.get());
                cam_info     = *info_msg;
                got_cam_info = true;
            }
        }
    }
    if (!got_cam_info)
        throw std::runtime_error(
            "No CameraInfo messages found on topic: " + info_topic);

    // ── Third pass: collect images and poses ──────────────────────────────
    struct RawFrame {
        uint64_t stamp_ns;
        cv::Mat  image;   // RGB
    };
    std::vector<RawFrame>    raw_frames;
    std::vector<StampedPose> poses;

    {
        rosbag2_cpp::Reader reader;
        reader.open(storage_opts);

        rclcpp::Serialization<sensor_msgs::msg::Image>            img_ser;
        rclcpp::Serialization<sensor_msgs::msg::CompressedImage>  cimg_ser;
        rclcpp::Serialization<geometry_msgs::msg::PoseStamped>    pose_ser;

        while (reader.has_next()) {
            auto msg = reader.read_next();

            // ── Image ──────────────────────────────────────────────────────
            if (msg->topic_name == img_topic) {
                RawFrame f;
                f.stamp_ns = static_cast<uint64_t>(msg->time_stamp);

                if (!is_compressed) {
                    rclcpp::SerializedMessage sm(*msg->serialized_data);
                    auto img_msg = std::make_shared<sensor_msgs::msg::Image>();
                    img_ser.deserialize_message(&sm, img_msg.get());
                    try {
                        f.image = rosImageToMat(*img_msg);
                    } catch (const std::exception &e) {
                        std::cerr << "[rosbag] Warning: skipping frame at t="
                                  << f.stamp_ns << ": " << e.what() << std::endl;
                        continue;
                    }
                } else {
                    rclcpp::SerializedMessage sm(*msg->serialized_data);
                    auto cimg_msg =
                        std::make_shared<sensor_msgs::msg::CompressedImage>();
                    cimg_ser.deserialize_message(&sm, cimg_msg.get());
                    std::vector<uint8_t> buf(cimg_msg->data.begin(),
                                             cimg_msg->data.end());
                    cv::Mat bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
                    if (bgr.empty()) {
                        std::cerr << "[rosbag] Warning: failed to decode "
                                     "compressed image at t="
                                  << f.stamp_ns << std::endl;
                        continue;
                    }
                    cv::cvtColor(bgr, f.image, cv::COLOR_BGR2RGB);
                }
                raw_frames.push_back(std::move(f));
            }

            // ── Pose ───────────────────────────────────────────────────────
            else if (has_pose_topic && msg->topic_name == pose_topic_used) {
                rclcpp::SerializedMessage sm(*msg->serialized_data);
                auto pose_msg =
                    std::make_shared<geometry_msgs::msg::PoseStamped>();
                pose_ser.deserialize_message(&sm, pose_msg.get());
                StampedPose p;
                p.stamp_ns = static_cast<uint64_t>(msg->time_stamp);
                p.tx = static_cast<float>(pose_msg->pose.position.x);
                p.ty = static_cast<float>(pose_msg->pose.position.y);
                p.tz = static_cast<float>(pose_msg->pose.position.z);
                p.qx = static_cast<float>(pose_msg->pose.orientation.x);
                p.qy = static_cast<float>(pose_msg->pose.orientation.y);
                p.qz = static_cast<float>(pose_msg->pose.orientation.z);
                p.qw = static_cast<float>(pose_msg->pose.orientation.w);
                poses.push_back(p);
            }
        }
    }

    if (raw_frames.empty())
        throw std::runtime_error(
            "No image frames found on topic: " + img_topic);

    std::cout << "[rosbag] Extracted " << raw_frames.size()
              << " frames from bag." << std::endl;

    // Sort poses by timestamp
    std::sort(poses.begin(), poses.end(),
              [](const StampedPose &a, const StampedPose &b) {
                  return a.stamp_ns < b.stamp_ns;
              });

    // ── Extract intrinsics ────────────────────────────────────────────────
    // K = [fx  0  cx]
    //     [ 0 fy  cy]
    //     [ 0  0   1]
    float fx = static_cast<float>(cam_info.k[0]);
    float fy = static_cast<float>(cam_info.k[4]);
    float cx = static_cast<float>(cam_info.k[2]);
    float cy = static_cast<float>(cam_info.k[5]);
    float k1 = cam_info.d.size() > 0 ? static_cast<float>(cam_info.d[0]) : 0.f;
    float k2 = cam_info.d.size() > 1 ? static_cast<float>(cam_info.d[1]) : 0.f;
    float p1 = cam_info.d.size() > 2 ? static_cast<float>(cam_info.d[2]) : 0.f;
    float p2 = cam_info.d.size() > 3 ? static_cast<float>(cam_info.d[3]) : 0.f;
    float k3 = cam_info.d.size() > 4 ? static_cast<float>(cam_info.d[4]) : 0.f;
    int   w  = static_cast<int>(cam_info.width);
    int   h  = static_cast<int>(cam_info.height);

    std::cout << "[rosbag] Streaming " << raw_frames.size()
              << " frames directly into memory (no disk I/O)." << std::endl;

    // ── Build InputData ───────────────────────────────────────────────────
    // Images are stored in Camera::preloadedImage so that Camera::loadImage()
    // can process them (scale, undistort, convert to tensor) without any
    // filesystem round-trip.  filePath is set to a logical identifier of the
    // form "bag_frame_<N>" (not a real path on disk).
    InputData result;
    result.scale       = 1.0f;
    result.translation = torch::zeros({3}, torch::kFloat32);
    result.points.xyz  = torch::zeros({0, 3}, torch::kFloat32);
    result.points.rgb  = torch::zeros({0, 3}, torch::kFloat32);

    for (size_t i = 0; i < raw_frames.size(); i++) {
        auto &frame = raw_frames[i];

        // Resolve camera-to-world pose
        torch::Tensor cam_to_world = torch::eye(4, torch::kFloat32);
        if (!poses.empty()) {
            const StampedPose *p = nearestPose(poses, frame.stamp_ns);
            cam_to_world = poseToMatrix(p->tx, p->ty, p->tz,
                                        p->qx, p->qy, p->qz, p->qw);
        }

        // Build Camera with a logical name – the image lives in memory, not on disk.
        Camera cam(w, h, fx, fy, cx, cy, k1, k2, k3, p1, p2,
                   cam_to_world,
                   /*filePath=*/"bag_frame_" + std::to_string(i));
        cam.id             = static_cast<int>(i);
        cam.preloadedImage = std::move(frame.image); // zero-copy hand-off
        result.cameras.push_back(std::move(cam));
    }

    return result;
}

} // namespace rb

#endif // HAVE_ROSBAG2
