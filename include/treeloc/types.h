#pragma once

#include <array>
#include <filesystem>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace treeloc {

namespace fs = std::filesystem;

struct Point {
    double x, y, z, qx, qy, qz, qw;
};

struct TreeData {
    Eigen::Matrix3d R;
    double location_x, location_y, location_z, dbh, dbh_approximation, score;
    int reconstructed, number_clusters;
};

struct KNNTriangles {
    std::vector<std::array<int, 3>> simplices;
};

struct TreeMatch {
    int query_idx;
    int cand_idx;
};

struct TransformationResult {
    Eigen::Matrix2d R;
    Eigen::Vector2d t;
    double overlap;
    std::vector<TreeMatch> matches;
};

struct PrecomputedData {
    Eigen::MatrixXd histogram;
    std::vector<Eigen::Vector2d> centers;
    KNNTriangles tri;
    std::vector<std::pair<long long, int>> hash_list;
    std::set<long long> hash_set;
    std::vector<TreeData> df;
    Eigen::Matrix4d scene_transform = Eigen::Matrix4d::Identity();
};

struct DatasetContext {
    fs::path dataset_root;
    std::vector<Point> trajectory_full;
};

}  // namespace treeloc
