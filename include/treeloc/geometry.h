#pragma once

#include <vector>

#include <Eigen/Dense>

#include "treeloc/types.h"

namespace treeloc {

Eigen::Matrix3d QuatToRotation(double qx, double qy, double qz, double qw);
Eigen::Matrix4d PoseRowToTransform(const Point& point);
Eigen::Matrix4d TreeDataToTransform(const TreeData& row);
TreeData ApplySceneTransform(const TreeData& row, const Eigen::Matrix4d& scene_transform);
Eigen::Matrix4d ComputeGlobalZAlignmentFallback(const std::vector<Eigen::Matrix4d>& tree_axes);
double WrapAngle(double angle);
void EulerZYX(const Eigen::Matrix3d& rotation, double& yaw, double& pitch, double& roll);
double RansacZOffset(const std::vector<double>& query_z,
                     const std::vector<double>& cand_z,
                     int n_iterations = 100,
                     size_t sample_size = 50,
                     double inlier_threshold = 0.3,
                     double min_inliers = 0.5);

}  // namespace treeloc
