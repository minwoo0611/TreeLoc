#pragma once

#include <vector>

#include <Eigen/Dense>

#include "treeloc/config.h"
#include "treeloc/types.h"

namespace treeloc {

double GetRadius(const TreeData& row);
int GetSpatialRange(const TreeData& row, const std::vector<RangeBin>& spatial_range_bins);
Eigen::MatrixXd ComputeHistogram(const std::vector<TreeData>& trees,
                                 const std::vector<RangeBin>& spatial_range_bins,
                                 const std::vector<RangeBin>& radius_bins);
double ChiSquaredDistance(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs);
KNNTriangles ComputeKnnTriangles(const std::vector<Eigen::Vector2d>& centers,
                                 int k,
                                 double min_dist,
                                 double max_dist);
std::vector<std::pair<long long, int>> GetTriangleHashes(const KNNTriangles& triangles,
                                                         const std::vector<Eigen::Vector2d>& centers,
                                                         double delta_l,
                                                         long long rho,
                                                         long long hash_modulus);
TransformationResult Compute2DTransformation(
    const std::vector<Eigen::Vector2d>& query_pts,
    const std::vector<Eigen::Vector2d>& cand_pts,
    const KNNTriangles& query_tri,
    const KNNTriangles& cand_tri,
    const std::vector<std::pair<long long, int>>& query_hashes,
    const std::vector<std::pair<long long, int>>& cand_hashes,
    const std::vector<TreeData>& query_df,
    const std::vector<TreeData>& cand_df);
PrecomputedData ProcessFile(int frame_idx,
                            const DatasetContext& context,
                            const Config& config,
                            const std::vector<RangeBin>& radius_bins);

}  // namespace treeloc
