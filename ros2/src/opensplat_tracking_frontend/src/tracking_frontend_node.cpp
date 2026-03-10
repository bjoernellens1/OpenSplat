/// tracking_frontend_node.cpp
///
/// RGB-D odometry frontend node for the OpenSplat live Gaussian mapper.
///
/// Subscriptions:
///   /camera/color/image       (sensor_msgs/Image, rgb8)
///   /camera/depth/image       (sensor_msgs/Image, 32FC1 metres)
///   /camera/color/camera_info (sensor_msgs/CameraInfo)
///
/// Publications:
///   /tracking/odometry            (nav_msgs/Odometry)
///   /tracking/status              (opensplat_live_interfaces/TrackingStatus)
///   /tracking/keyframe_candidate  (opensplat_live_interfaces/Keyframe)
///
/// The baseline frontend uses OpenCV's frame-to-frame photometric + geometric
/// residual minimisation (simplified iterative depth alignment).  The class
/// FrontendInterface defines the API so the backend can be swapped without
/// touching downstream nodes.
///
/// V1 limitations:
///   - No loop-closure detection
///   - No relocalization
///   - Tracking degrades in textureless regions

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "opensplat_live_interfaces/msg/keyframe.hpp"
#include "opensplat_live_interfaces/msg/tracking_status.hpp"

using namespace std::chrono_literals;

// ─── Simple SE(3) pose (col-major 4×4) ────────────────────────────────────────
struct Pose4x4 {
    double m[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    cv::Matx44d mat() const {
        return cv::Matx44d(m[0],m[4],m[8],m[12],
                           m[1],m[5],m[9],m[13],
                           m[2],m[6],m[10],m[14],
                           m[3],m[7],m[11],m[15]);
    }

    // Extract quaternion (w,x,y,z) and translation
    void getPoseComponents(float &tx, float &ty, float &tz,
                           float &qx, float &qy, float &qz, float &qw) const {
        tx = static_cast<float>(m[12]);
        ty = static_cast<float>(m[13]);
        tz = static_cast<float>(m[14]);

        // Rotation matrix elements (row-major in col-major layout)
        double r00=m[0], r10=m[1], r20=m[2];
        double r01=m[4], r11=m[5], r21=m[6];
        double r02=m[8], r12=m[9], r22=m[10];
        double trace = r00+r11+r22;
        double s;
        if (trace > 0.0) {
            s = 0.5 / std::sqrt(trace + 1.0);
            qw = 0.25f / s;
            qx = (r21-r12)*s;
            qy = (r02-r20)*s;
            qz = (r10-r01)*s;
        } else if (r00 > r11 && r00 > r22) {
            s = 2.0 * std::sqrt(1.0 + r00 - r11 - r22);
            qw = (r21-r12)/s;
            qx = 0.25*s;
            qy = (r01+r10)/s;
            qz = (r02+r20)/s;
        } else if (r11 > r22) {
            s = 2.0 * std::sqrt(1.0 + r11 - r00 - r22);
            qw = (r02-r20)/s;
            qx = (r01+r10)/s;
            qy = 0.25*s;
            qz = (r12+r21)/s;
        } else {
            s = 2.0 * std::sqrt(1.0 + r22 - r00 - r11);
            qw = (r10-r01)/s;
            qx = (r02+r20)/s;
            qy = (r12+r21)/s;
            qz = 0.25*s;
        }
        qx = static_cast<float>(qx);
        qy = static_cast<float>(qy);
        qz = static_cast<float>(qz);
        qw = static_cast<float>(qw);
    }
};

// ─── Minimal RGB-D frame-to-frame tracker ────────────────────────────────────
class RgbdOdometry {
public:
    explicit RgbdOdometry(float fx, float fy, float cx, float cy,
                          int levels = 3)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), levels_(levels) {}

    void updateIntrinsics(float fx, float fy, float cx, float cy) {
        fx_=fx; fy_=fy; cx_=cx; cy_=cy;
    }

    /// Estimate the relative pose delta between the reference and current frame.
    /// Returns a normalised confidence score in [0,1] and updates worldPose_.
    float track(const cv::Mat &grayRef, const cv::Mat &depthRef,
                const cv::Mat &grayCur, const cv::Mat &depthCur,
                cv::Matx44d &relPose)
    {
        // Build image pyramids
        std::vector<cv::Mat> pyrGrayRef(levels_), pyrGrayCur(levels_);
        std::vector<cv::Mat> pyrDepthRef(levels_);
        pyrGrayRef[0]  = grayRef;
        pyrGrayCur[0]  = grayCur;
        pyrDepthRef[0] = depthRef;
        for (int l = 1; l < levels_; ++l) {
            cv::pyrDown(pyrGrayRef[l-1], pyrGrayRef[l]);
            cv::pyrDown(pyrGrayCur[l-1], pyrGrayCur[l]);
            cv::resize(pyrDepthRef[l-1], pyrDepthRef[l],
                       pyrGrayRef[l].size(), 0, 0, cv::INTER_NEAREST);
        }

        // Start from identity
        cv::Matx44d T = cv::Matx44d::eye();

        for (int l = levels_-1; l >= 0; --l) {
            float scale = std::pow(0.5f, l);
            float fxl = fx_*scale, fyl = fy_*scale;
            float cxl = cx_*scale, cyl = cy_*scale;
            T = refinePyramidLevel(pyrGrayRef[l], pyrDepthRef[l],
                                   pyrGrayCur[l], fxl, fyl, cxl, cyl, T);
        }
        relPose = T;

        // Confidence based on reprojection residual (coarse heuristic)
        float residual = computeResidual(grayRef, depthRef, grayCur, T);
        float confidence = std::exp(-residual / 0.1f);
        return std::clamp(confidence, 0.0f, 1.0f);
    }

private:
    float fx_, fy_, cx_, cy_;
    int   levels_;

    cv::Matx44d refinePyramidLevel(
        const cv::Mat &grayRef, const cv::Mat &depthRef,
        const cv::Mat &grayCur,
        float fx, float fy, float cx, float cy,
        const cv::Matx44d &initT)
    {
        cv::Matx44d T = initT;
        const int MAX_ITER = 10;

        for (int iter = 0; iter < MAX_ITER; ++iter) {
            // Collect valid reference pixels
            cv::Mat Jt(0, 6, CV_64F);
            cv::Mat residuals(0, 1, CV_64F);

            int H = grayRef.rows, W = grayRef.cols;
            for (int v = 1; v < H-1; v += 2) {
                for (int u = 1; u < W-1; u += 2) {
                    float d = depthRef.at<float>(v, u);
                    if (d <= 0.01f || d > 8.0f) continue;

                    // Back-project reference pixel to 3-D
                    double Xr = (u - cx) * d / fx;
                    double Yr = (v - cy) * d / fy;
                    double Zr = d;

                    // Transform to current frame
                    double Xc = T(0,0)*Xr + T(0,1)*Yr + T(0,2)*Zr + T(0,3);
                    double Yc = T(1,0)*Xr + T(1,1)*Yr + T(1,2)*Zr + T(1,3);
                    double Zc = T(2,0)*Xr + T(2,1)*Yr + T(2,2)*Zr + T(2,3);
                    if (Zc <= 0.01) continue;

                    double uc = fx * Xc / Zc + cx;
                    double vc = fy * Yc / Zc + cy;
                    if (uc < 1 || uc >= W-1 || vc < 1 || vc >= H-1) continue;

                    // Bilinear interpolation in current frame
                    int ui = static_cast<int>(uc), vi = static_cast<int>(vc);
                    double dx = uc - ui, dy = vc - vi;
                    double Ic = (1-dy)*((1-dx)*grayCur.at<uint8_t>(vi,ui)
                                       +dx*grayCur.at<uint8_t>(vi,ui+1))
                              + dy*((1-dx)*grayCur.at<uint8_t>(vi+1,ui)
                                    +dx*grayCur.at<uint8_t>(vi+1,ui+1));

                    double Ir = grayRef.at<uint8_t>(v, u);
                    double r  = (Ic - Ir) / 255.0;

                    // Image gradient at projected location
                    double gx = (grayCur.at<uint8_t>(vi, ui+1)
                               - grayCur.at<uint8_t>(vi, ui-1)) / (2.0 * 255.0);
                    double gy = (grayCur.at<uint8_t>(vi+1, ui)
                               - grayCur.at<uint8_t>(vi-1, ui)) / (2.0 * 255.0);

                    // Jacobian row for SE(3) twist [tx ty tz rx ry rz]
                    double invZ = 1.0 / Zc;
                    double J[6] = {
                        gx * fx * invZ,
                        gy * fy * invZ,
                        -(gx * fx * Xc + gy * fy * Yc) * invZ * invZ,
                        -(gx * fx * Xc * Yc * invZ * invZ + gy * fy * (1 + Yc*Yc * invZ*invZ)),
                         (gx * fx * (1 + Xc*Xc * invZ*invZ) + gy * fy * Xc * Yc * invZ * invZ),
                        (-gx * fx * Yc + gy * fy * Xc) * invZ
                    };
                    cv::Mat jRow(1, 6, CV_64F, J);
                    Jt.push_back(jRow);
                    residuals.push_back(cv::Mat(1, 1, CV_64F, &r));
                }
            }
            if (residuals.rows < 10) break;

            // Solve (J^T J) dx = -J^T r
            cv::Mat JtJ = Jt.t() * Jt;
            cv::Mat Jtr = Jt.t() * residuals;
            cv::Mat dx;
            cv::solve(JtJ, -Jtr, dx, cv::DECOMP_CHOLESKY);

            // Apply twist update using the standard Rodrigues formula.
            // dT = exp([ω] * θ) with unit axis n and angle θ:
            //   R = I + sin(θ)*K + (1-cos(θ))*K²   (K is skew of unit n)
            //   R[0][0] = 1 - (1-cos(θ))*(ny²+nz²), etc.
            double tx=dx.at<double>(0), ty=dx.at<double>(1), tz=dx.at<double>(2);
            double rx=dx.at<double>(3), ry=dx.at<double>(4), rz=dx.at<double>(5);
            double angle = std::sqrt(rx*rx + ry*ry + rz*rz);
            cv::Matx44d dT = cv::Matx44d::eye();
            if (angle > 1e-10) {
                double s = std::sin(angle);
                double c = 1.0 - std::cos(angle);
                double nx=rx/angle, ny=ry/angle, nz=rz/angle;
                dT(0,0)=1-c*(ny*ny+nz*nz); dT(0,1)=c*nx*ny-s*nz; dT(0,2)=c*nx*nz+s*ny;
                dT(1,0)=c*nx*ny+s*nz; dT(1,1)=1-c*(nx*nx+nz*nz); dT(1,2)=c*ny*nz-s*nx;
                dT(2,0)=c*nx*nz-s*ny; dT(2,1)=c*ny*nz+s*nx; dT(2,2)=1-c*(nx*nx+ny*ny);
            }
            dT(0,3)=tx; dT(1,3)=ty; dT(2,3)=tz;
            T = dT * T;

            double norm = cv::norm(dx);
            if (norm < 1e-6) break;
        }
        return T;
    }

    float computeResidual(const cv::Mat &grayRef, const cv::Mat &depthRef,
                          const cv::Mat &grayCur, const cv::Matx44d &T)
    {
        int H = grayRef.rows, W = grayRef.cols;
        double sumR = 0; int count = 0;
        for (int v = 0; v < H; v += 4) {
            for (int u = 0; u < W; u += 4) {
                float d = depthRef.at<float>(v, u);
                if (d <= 0) continue;
                double Xc = T(0,0)*(u-cx_)*d/fx_ + T(0,1)*(v-cy_)*d/fy_ + T(0,2)*d + T(0,3);
                double Yc = T(1,0)*(u-cx_)*d/fx_ + T(1,1)*(v-cy_)*d/fy_ + T(1,2)*d + T(1,3);
                double Zc = T(2,0)*(u-cx_)*d/fx_ + T(2,1)*(v-cy_)*d/fy_ + T(2,2)*d + T(2,3);
                if (Zc <= 0) continue;
                double uc = fx_*Xc/Zc+cx_, vc = fy_*Yc/Zc+cy_;
                if (uc < 0 || uc >= W-1 || vc < 0 || vc >= H-1) continue;
                int ui = static_cast<int>(uc), vi = static_cast<int>(vc);
                double Ic = grayCur.at<uint8_t>(vi, ui);
                double Ir = grayRef.at<uint8_t>(v, u);
                sumR += std::abs(Ic - Ir) / 255.0;
                ++count;
            }
        }
        return count > 0 ? static_cast<float>(sumR / count) : 1.0f;
    }
};

// ─── ROS 2 Node ───────────────────────────────────────────────────────────────
class TrackingFrontendNode : public rclcpp::Node {
public:
    TrackingFrontendNode()
        : Node("opensplat_tracking_frontend")
    {
        this->declare_parameter("kf_trans_thresh",   0.10f);
        this->declare_parameter("kf_rot_thresh_deg", 7.0f);
        this->declare_parameter("kf_max_interval_sec", 0.5f);
        this->declare_parameter("tracker_levels", 3);

        kfTransThr_ = get_parameter("kf_trans_thresh").as_double();
        kfRotThr_   = get_parameter("kf_rot_thresh_deg").as_double() * M_PI / 180.0;
        kfMaxDt_    = get_parameter("kf_max_interval_sec").as_double();

        auto qos = rclcpp::QoS(rclcpp::KeepLast(5));
        subRgb_   = create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image", qos,
            std::bind(&TrackingFrontendNode::rgbCallback, this, std::placeholders::_1));
        subDepth_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image", qos,
            std::bind(&TrackingFrontendNode::depthCallback, this, std::placeholders::_1));
        subInfo_  = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info", qos,
            std::bind(&TrackingFrontendNode::infoCallback, this, std::placeholders::_1));

        pubOdom_      = create_publisher<nav_msgs::msg::Odometry>("/tracking/odometry", qos);
        pubStatus_    = create_publisher<opensplat_live_interfaces::msg::TrackingStatus>(
            "/tracking/status", qos);
        pubKeyframe_  = create_publisher<opensplat_live_interfaces::msg::Keyframe>(
            "/tracking/keyframe_candidate", qos);

        RCLCPP_INFO(get_logger(), "TrackingFrontendNode ready");
    }

private:
    // ── Latest buffered frames ────────────────────────────────────────────
    sensor_msgs::msg::Image::SharedPtr latestRgb_, latestDepth_;
    std::mutex frameMutex_;

    sensor_msgs::msg::CameraInfo::SharedPtr camInfo_;
    bool gotInfo_ = false;

    // Reference frame for odometry
    cv::Mat refGray_, refDepth_;
    cv::Matx44d worldPose_ = cv::Matx44d::eye();
    cv::Matx44d kfPose_    = cv::Matx44d::eye();
    double lastKfTime_     = -1.0;
    bool   hasRef_         = false;

    std::unique_ptr<RgbdOdometry> tracker_;

    float fx_=525,fy_=525,cx_=320,cy_=240;

    float kfTransThr_, kfRotThr_, kfMaxDt_;
    uint32_t kfCount_ = 0;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subRgb_, subDepth_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr subInfo_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdom_;
    rclcpp::Publisher<opensplat_live_interfaces::msg::TrackingStatus>::SharedPtr pubStatus_;
    rclcpp::Publisher<opensplat_live_interfaces::msg::Keyframe>::SharedPtr pubKeyframe_;

    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        if (gotInfo_) return;
        camInfo_ = msg;
        fx_ = static_cast<float>(msg->k[0]);
        fy_ = static_cast<float>(msg->k[4]);
        cx_ = static_cast<float>(msg->k[2]);
        cy_ = static_cast<float>(msg->k[5]);
        gotInfo_ = true;
        RCLCPP_INFO(get_logger(), "Camera intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                    fx_, fy_, cx_, cy_);
    }

    void rgbCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(frameMutex_);
        latestRgb_ = msg;
        tryTrack();
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(frameMutex_);
        latestDepth_ = msg;
        tryTrack();
    }

    void tryTrack() {
        if (!latestRgb_ || !latestDepth_ || !gotInfo_) return;

        // Simple timestamp matching: accept if within 50 ms
        auto tRgb   = rclcpp::Time(latestRgb_->header.stamp);
        auto tDepth = rclcpp::Time(latestDepth_->header.stamp);
        double dtMs = std::abs((tRgb - tDepth).seconds()) * 1000.0;
        if (dtMs > 50.0) return;

        // Convert to OpenCV
        cv::Mat gray, depth32f;
        try {
            auto cvRgb   = cv_bridge::toCvShare(latestRgb_,   "rgb8");
            auto cvDepth = cv_bridge::toCvShare(latestDepth_, "32FC1");
            cv::cvtColor(cvRgb->image, gray, cv::COLOR_RGB2GRAY);
            depth32f = cvDepth->image.clone();
        } catch (const cv_bridge::Exception &e) {
            RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
            return;
        }

        // Initialise tracker
        if (!tracker_)
            tracker_ = std::make_unique<RgbdOdometry>(fx_, fy_, cx_, cy_);
        else
            tracker_->updateIntrinsics(fx_, fy_, cx_, cy_);

        double timestamp = tRgb.seconds();

        float confidence = 1.0f;
        if (!hasRef_) {
            refGray_  = gray.clone();
            refDepth_ = depth32f.clone();
            worldPose_ = cv::Matx44d::eye();
            kfPose_    = worldPose_;
            hasRef_    = true;
            lastKfTime_ = timestamp;
        } else {
            cv::Matx44d relPose;
            confidence = tracker_->track(refGray_, refDepth_, gray, depth32f, relPose);
            worldPose_ = relPose * worldPose_;
            // Update reference continuously (frame-to-frame)
            refGray_  = gray.clone();
            refDepth_ = depth32f.clone();
        }

        // Publish odometry
        publishOdometry(latestRgb_->header, worldPose_, confidence);

        // Keyframe decision
        bool isKf = false;
        double dt = timestamp - lastKfTime_;
        // Translation difference
        double dx = worldPose_(0,3)-kfPose_(0,3);
        double dy = worldPose_(1,3)-kfPose_(1,3);
        double dz = worldPose_(2,3)-kfPose_(2,3);
        double trans = std::sqrt(dx*dx+dy*dy+dz*dz);
        if (trans > kfTransThr_) isKf = true;
        if (dt > kfMaxDt_)       isKf = true;
        if (confidence < 0.3f)   isKf = true;

        auto statusMsg = opensplat_live_interfaces::msg::TrackingStatus();
        statusMsg.header = latestRgb_->header;
        statusMsg.confidence = confidence;
        statusMsg.is_keyframe = isKf;
        statusMsg.status = confidence > 0.5f
            ? opensplat_live_interfaces::msg::TrackingStatus::STATUS_OK
            : opensplat_live_interfaces::msg::TrackingStatus::STATUS_DEGRADED;
        pubStatus_->publish(statusMsg);

        if (isKf && camInfo_) {
            publishKeyframe(latestRgb_, latestDepth_, *camInfo_, worldPose_, confidence);
            kfPose_ = worldPose_;
            lastKfTime_ = timestamp;
        }

        latestRgb_.reset();
        latestDepth_.reset();
    }

    void publishOdometry(const std_msgs::msg::Header &hdr,
                         const cv::Matx44d &pose, float /*confidence*/)
    {
        nav_msgs::msg::Odometry odom;
        odom.header = hdr;
        odom.header.frame_id = "map";
        odom.child_frame_id  = "camera";
        odom.pose.pose.position.x = pose(0,3);
        odom.pose.pose.position.y = pose(1,3);
        odom.pose.pose.position.z = pose(2,3);

        // Rotation → quaternion
        double trace = pose(0,0)+pose(1,1)+pose(2,2);
        double qw, qx, qy, qz;
        if (trace > 0) {
            double s = 0.5/std::sqrt(trace+1);
            qw=0.25/s; qx=(pose(2,1)-pose(1,2))*s;
            qy=(pose(0,2)-pose(2,0))*s; qz=(pose(1,0)-pose(0,1))*s;
        } else {
            qw=0; qx=0; qy=0; qz=1;
        }
        odom.pose.pose.orientation.w = qw;
        odom.pose.pose.orientation.x = qx;
        odom.pose.pose.orientation.y = qy;
        odom.pose.pose.orientation.z = qz;
        pubOdom_->publish(odom);
    }

    void publishKeyframe(
        const sensor_msgs::msg::Image::SharedPtr &rgb,
        const sensor_msgs::msg::Image::SharedPtr &depth,
        const sensor_msgs::msg::CameraInfo &info,
        const cv::Matx44d &pose, float /*confidence*/)
    {
        opensplat_live_interfaces::msg::Keyframe kf;
        kf.header       = rgb->header;
        kf.keyframe_id  = ++kfCount_;
        kf.rgb          = *rgb;
        kf.depth        = *depth;
        kf.camera_info  = info;

        kf.pose.header  = rgb->header;
        kf.pose.pose.position.x = pose(0,3);
        kf.pose.pose.position.y = pose(1,3);
        kf.pose.pose.position.z = pose(2,3);

        double trace = pose(0,0)+pose(1,1)+pose(2,2);
        double qw, qx, qy, qz;
        if (trace > 0) {
            double s = 0.5/std::sqrt(trace+1);
            qw=0.25/s; qx=(pose(2,1)-pose(1,2))*s;
            qy=(pose(0,2)-pose(2,0))*s; qz=(pose(1,0)-pose(0,1))*s;
        } else {
            qw=0; qx=0; qy=0; qz=1;
        }
        kf.pose.pose.orientation.w = qw;
        kf.pose.pose.orientation.x = qx;
        kf.pose.pose.orientation.y = qy;
        kf.pose.pose.orientation.z = qz;

        pubKeyframe_->publish(kf);
        RCLCPP_DEBUG(get_logger(), "Keyframe #%u published", kfCount_);
    }
};

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrackingFrontendNode>());
    rclcpp::shutdown();
    return 0;
}
