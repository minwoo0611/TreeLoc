#include "treeloc/geometry.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace treeloc {

Eigen::Matrix3d QuatToRotation(double qx, double qy, double qz, double qw) {
    Eigen::Quaterniond quaternion(qw, qx, qy, qz);
    return quaternion.toRotationMatrix();
}

Eigen::Matrix4d PoseRowToTransform(const Point& point) {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 1>(0, 3) = Eigen::Vector3d(point.x, point.y, point.z);
    transform.block<3, 3>(0, 0) = QuatToRotation(point.qx, point.qy, point.qz, point.qw);
    return transform;
}

Eigen::Matrix4d TreeDataToTransform(const TreeData& row) {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = row.R;
    transform.block<3, 1>(0, 3) =
        Eigen::Vector3d(row.location_x, row.location_y, row.location_z);
    return transform;
}

TreeData ApplySceneTransform(const TreeData& row, const Eigen::Matrix4d& scene_transform) {
    TreeData transformed = row;
    Eigen::Matrix4d transform = scene_transform * TreeDataToTransform(row);
    transformed.R = transform.block<3, 3>(0, 0);
    transformed.location_x = transform(0, 3);
    transformed.location_y = transform(1, 3);
    transformed.location_z = transform(2, 3);
    return transformed;
}

Eigen::Matrix4d ComputeGlobalZAlignmentFallback(const std::vector<Eigen::Matrix4d>& tree_axes) {
    std::vector<Eigen::Vector3d> principal_axes;
    principal_axes.reserve(tree_axes.size());

    for (const auto& axis_transform : tree_axes) {
        Eigen::Vector3d axis = axis_transform.block<3, 1>(0, 2);
        if (!axis.allFinite() || axis.norm() < 1e-9) continue;
        principal_axes.push_back(axis.normalized());
    }

    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    if (principal_axes.empty()) {
        return transform;
    }

    Eigen::Vector3d combined_axis = Eigen::Vector3d::Zero();
    for (const auto& axis : principal_axes) {
        combined_axis += axis;
    }

    if (!combined_axis.allFinite() || combined_axis.norm() < 1e-9) {
        return transform;
    }

    combined_axis.normalize();
    if (combined_axis.z() < 0.0) {
        combined_axis = -combined_axis;
    }

    const Eigen::Vector3d target_axis(0.0, 0.0, 1.0);
    const double c = std::clamp(combined_axis.dot(target_axis), -1.0, 1.0);

    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    if (std::abs(c - 1.0) < 1e-9) {
        rotation_matrix.setIdentity();
    } else if (std::abs(c + 1.0) < 1e-9) {
        rotation_matrix << -1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, -1.0;
    } else {
        const Eigen::Vector3d v = combined_axis.cross(target_axis);
        Eigen::Matrix3d v_skew;
        v_skew << 0.0, -v.z(), v.y(),
                  v.z(), 0.0, -v.x(),
                  -v.y(), v.x(), 0.0;
        rotation_matrix =
            Eigen::Matrix3d::Identity() + v_skew + v_skew * v_skew * (1.0 / (1.0 + c));
    }

    transform.block<3, 3>(0, 0) = rotation_matrix;
    return transform;
}

double WrapAngle(double angle) {
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0.0) angle += 2.0 * M_PI;
    return angle - M_PI;
}

void EulerZYX(const Eigen::Matrix3d& rotation, double& yaw, double& pitch, double& roll) {
    yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    pitch = std::atan2(
        -rotation(2, 0),
        std::sqrt(rotation(2, 1) * rotation(2, 1) + rotation(2, 2) * rotation(2, 2)));
    roll = std::atan2(rotation(2, 1), rotation(2, 2));
}

double RansacZOffset(const std::vector<double>& query_z,
                     const std::vector<double>& cand_z,
                     int n_iterations,
                     size_t sample_size,
                     double inlier_threshold,
                     double min_inliers) {
    const size_t n_points = query_z.size();
    if (n_points != cand_z.size()) {
        throw std::invalid_argument("query_z and cand_z must have the same size");
    }
    if (n_points < sample_size) {
        throw std::invalid_argument("Not enough points for sampling");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, n_points - 1);

    size_t best_inlier_count = 0;
    double best_z_offset = 0.0;

    for (int i = 0; i < n_iterations; ++i) {
        std::vector<size_t> sample_indices(sample_size);
        for (size_t j = 0; j < sample_size; ++j) {
            sample_indices[j] = dist(gen);
        }

        double sample_z_offset = 0.0;
        for (size_t idx : sample_indices) {
            sample_z_offset += (query_z[idx] - cand_z[idx]);
        }
        sample_z_offset /= sample_size;

        size_t inlier_count = 0;
        std::vector<double> inlier_diffs;
        for (size_t j = 0; j < n_points; ++j) {
            const double z_diff = query_z[j] - cand_z[j];
            if (std::abs(z_diff - sample_z_offset) < inlier_threshold) {
                ++inlier_count;
                inlier_diffs.push_back(z_diff);
            }
        }

        if (inlier_count > best_inlier_count && inlier_count >= min_inliers * n_points) {
            best_z_offset =
                std::accumulate(inlier_diffs.begin(), inlier_diffs.end(), 0.0) /
                inlier_diffs.size();
            best_inlier_count = inlier_count;
        }
    }

    if (best_inlier_count == 0) {
        double sum_diff = 0.0;
        for (size_t i = 0; i < n_points; ++i) {
            sum_diff += (query_z[i] - cand_z[i]);
        }
        best_z_offset = sum_diff / n_points;
    }

    return best_z_offset;
}

}  // namespace treeloc
