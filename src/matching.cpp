#include "treeloc/matching.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nanoflann.hpp>

#include "treeloc/geometry.h"
#include "treeloc/io.h"

namespace treeloc {

namespace {

std::tuple<double, double, double> TriangleSideLengths(
    const std::vector<Eigen::Vector2d>& points,
    const std::array<int, 3>& simplex) {
    const Eigen::Vector2d& p1 = points[simplex[0]];
    const Eigen::Vector2d& p2 = points[simplex[1]];
    const Eigen::Vector2d& p3 = points[simplex[2]];

    std::array<double, 3> lengths = {
        (p1 - p2).norm(),
        (p2 - p3).norm(),
        (p3 - p1).norm()
    };
    std::sort(lengths.begin(), lengths.end());
    return {lengths[0], lengths[1], lengths[2]};
}

long long TriangleHash(double a, double b, double c, double delta_l, long long rho,
                       long long hash_modulus) {
    const int ell_1 = static_cast<int>(std::round(a / delta_l));
    const int ell_2 = static_cast<int>(std::round(b / delta_l));
    const int ell_3 = static_cast<int>(std::round(c / delta_l));

    long long hash = (ell_3 * rho + ell_2) % hash_modulus;
    hash = (hash * rho + ell_1) % hash_modulus;
    return hash;
}

std::vector<TreeData> SelectTreesForDescriptors(const std::vector<TreeData>& trees,
                                                int required_clusters) {
    std::vector<TreeData> non_reconstructed;
    std::vector<TreeData> filtered;
    filtered.reserve(trees.size());

    for (const auto& row : trees) {
        if (row.reconstructed == 1) {
            filtered.push_back(row);
        } else if (row.number_clusters > 2 && row.score > 0.1) {
            non_reconstructed.push_back(row);
        }
    }

    if (static_cast<int>(filtered.size()) < required_clusters) {
        const int need_count = required_clusters - static_cast<int>(filtered.size());
        std::sort(non_reconstructed.begin(), non_reconstructed.end(),
                  [](const TreeData& lhs, const TreeData& rhs) { return lhs.score > rhs.score; });
        for (int i = 0; i < need_count && i < static_cast<int>(non_reconstructed.size()); ++i) {
            filtered.push_back(non_reconstructed[i]);
        }
    }

    if (static_cast<int>(filtered.size()) < required_clusters) {
        std::vector<TreeData> supplemental;
        for (const auto& row : trees) {
            if (row.reconstructed != 1 && row.number_clusters == 2 && row.score > 0.1) {
                supplemental.push_back(row);
            }
        }
        std::sort(supplemental.begin(), supplemental.end(),
                  [](const TreeData& lhs, const TreeData& rhs) { return lhs.score > rhs.score; });
        const int need_count = required_clusters - static_cast<int>(filtered.size());
        for (int i = 0; i < need_count && i < static_cast<int>(supplemental.size()); ++i) {
            filtered.push_back(supplemental[i]);
        }
    }

    std::vector<TreeData> final_trees;
    final_trees.reserve(filtered.size());
    for (const auto& row : filtered) {
        if (row.location_x >= -30.0 && row.location_x <= 30.0 &&
            row.location_y >= -30.0 && row.location_y <= 30.0) {
            final_trees.push_back(row);
        }
    }
    return final_trees;
}

}  // namespace

std::vector<RangeBin> BuildRadiusBins(const Config& config) {
    std::vector<RangeBin> radius_bins;
    if (config.total_section != 1) {
        const double step_size =
            (config.max_radius - config.min_radius) / (config.total_section + 1);
        for (int i = 0; i < config.total_section - 1; ++i) {
            radius_bins.emplace_back(
                config.min_radius + i * step_size,
                config.min_radius + i * step_size + config.bin_width);
        }
        radius_bins.emplace_back(radius_bins.back().first + step_size,
                                 std::numeric_limits<double>::infinity());
    } else {
        radius_bins.emplace_back(0.0, std::numeric_limits<double>::infinity());
    }
    return radius_bins;
}

double GetRadius(const TreeData& row) {
    return std::isnan(row.dbh) ? row.dbh_approximation : row.dbh;
}

int GetSpatialRange(const TreeData& row, const std::vector<RangeBin>& spatial_range_bins) {
    const double distance = std::sqrt(row.location_x * row.location_x +
                                      row.location_y * row.location_y);
    for (size_t i = 0; i < spatial_range_bins.size(); ++i) {
        if (spatial_range_bins[i].first < distance &&
            distance <= spatial_range_bins[i].second) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

Eigen::MatrixXd ComputeHistogram(const std::vector<TreeData>& trees,
                                 const std::vector<RangeBin>& spatial_range_bins,
                                 const std::vector<RangeBin>& radius_bins) {
    Eigen::MatrixXd histogram(spatial_range_bins.size(), radius_bins.size());
    histogram.setZero();
    for (const auto& row : trees) {
        const double radius = GetRadius(row);
        const int spatial_bin = GetSpatialRange(row, spatial_range_bins);
        if (spatial_bin == -1) continue;

        for (size_t j = 0; j < radius_bins.size(); ++j) {
            if ((radius_bins[j].first < radius && radius <= radius_bins[j].second) ||
                (j == radius_bins.size() - 1 && radius > radius_bins[j].first)) {
                histogram(spatial_bin, j) += 1.0;
                break;
            }
        }
    }
    return histogram;
}

double ChiSquaredDistance(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs) {
    double sum = 0.0;
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            const double diff = lhs(i, j) - rhs(i, j);
            sum += diff * diff / (lhs(i, j) + rhs(i, j) + 1e-10);
        }
    }
    return sum;
}

KNNTriangles ComputeKnnTriangles(const std::vector<Eigen::Vector2d>& centers,
                                 int k,
                                 double min_dist,
                                 double max_dist) {
    auto order_by_edge_length = [&](int i, int j, int k_idx) -> std::array<int, 3> {
        const double d_ij = (centers[i] - centers[j]).squaredNorm();
        const double d_jk = (centers[j] - centers[k_idx]).squaredNorm();
        const double d_ki = (centers[k_idx] - centers[i]).squaredNorm();

        int a = i;
        int b = j;
        int c = k_idx;
        double dab = d_ij;
        double dbc = d_jk;
        double dca = d_ki;

        if (d_jk >= d_ij && d_jk >= d_ki) {
            a = j;
            b = k_idx;
            c = i;
            dab = d_jk;
            dbc = d_ki;
            dca = d_ij;
        } else if (d_ki >= d_ij && d_ki >= d_jk) {
            a = k_idx;
            b = i;
            c = j;
            dab = d_ki;
            dbc = d_ij;
            dca = d_jk;
        }

        if (dbc < dca) {
            std::swap(a, b);
            dab = (centers[a] - centers[b]).squaredNorm();
            dbc = (centers[b] - centers[c]).squaredNorm();
            dca = (centers[c] - centers[a]).squaredNorm();
            (void)dab;
        }
        return {a, b, c};
    };

    KNNTriangles triangles;
    if (centers.size() < 3) return triangles;

    Eigen::MatrixXd points(centers.size(), 2);
    for (size_t i = 0; i < centers.size(); ++i) {
        points.row(i) = centers[i];
    }

    using KDTree =
        nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd, 2,
                                            nanoflann::metric_L2_Simple>;
    KDTree tree(2, std::ref(points), 10);
    tree.index->buildIndex();

    std::set<std::array<int, 3>> triples;
    std::vector<size_t> idx(k + 1);
    std::vector<double> d2(k + 1);

    for (size_t i = 0; i < centers.size(); ++i) {
        nanoflann::KNNResultSet<double> result_set(k + 1);
        result_set.init(idx.data(), d2.data());
        tree.index->findNeighbors(result_set, centers[i].data(), {});

        std::vector<int> neighbors;
        for (size_t j = 1; j < result_set.size(); ++j) {
            const double distance = std::sqrt(d2[j]);
            if (distance >= min_dist && distance <= max_dist) {
                neighbors.push_back(static_cast<int>(idx[j]));
            }
        }
        std::sort(neighbors.begin(), neighbors.end());

        for (size_t a = 0; a + 2 < neighbors.size(); ++a) {
            for (size_t b = a + 1; b + 1 < neighbors.size(); ++b) {
                for (size_t c = b + 1; c < neighbors.size(); ++c) {
                    triples.insert(order_by_edge_length(
                        neighbors[a], neighbors[b], neighbors[c]));
                }
            }
        }
    }

    triangles.simplices.assign(triples.begin(), triples.end());
    return triangles;
}

std::vector<std::pair<long long, int>> GetTriangleHashes(
    const KNNTriangles& triangles,
    const std::vector<Eigen::Vector2d>& centers,
    double delta_l,
    long long rho,
    long long hash_modulus) {
    std::vector<std::pair<long long, int>> hashes;
    hashes.reserve(triangles.simplices.size());
    for (size_t simplex_idx = 0; simplex_idx < triangles.simplices.size(); ++simplex_idx) {
        auto [a, b, c] = TriangleSideLengths(centers, triangles.simplices[simplex_idx]);
        const double s = (a + b + c) / 2.0;
        const double area =
            std::sqrt(std::max(s * (s - a) * (s - b) * (s - c), 0.0));
        long long hash_value = TriangleHash(a, b, c, delta_l, rho, hash_modulus);
        const int area_bucket = static_cast<int>(std::round(area / delta_l));
        hash_value = (hash_value * rho + area_bucket) % hash_modulus;
        hashes.emplace_back(hash_value, static_cast<int>(simplex_idx));
    }
    return hashes;
}

TransformationResult Compute2DTransformation(
    const std::vector<Eigen::Vector2d>& query_pts,
    const std::vector<Eigen::Vector2d>& cand_pts,
    const KNNTriangles& query_tri,
    const KNNTriangles& cand_tri,
    const std::vector<std::pair<long long, int>>& query_hashes,
    const std::vector<std::pair<long long, int>>& cand_hashes,
    const std::vector<TreeData>& query_df,
    const std::vector<TreeData>& cand_df) {
    TransformationResult result;
    result.R.setIdentity();
    result.t.setZero();
    result.overlap = 0.0;

    std::unordered_map<long long, int> query_hash_to_index;
    std::unordered_map<long long, int> cand_hash_to_index;
    for (const auto& [hash, idx] : query_hashes) query_hash_to_index[hash] = idx;
    for (const auto& [hash, idx] : cand_hashes) cand_hash_to_index[hash] = idx;

    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> correspondences;
    for (const auto& [hash, q_idx] : query_hash_to_index) {
        auto it = cand_hash_to_index.find(hash);
        if (it == cand_hash_to_index.end()) continue;

        const auto& q_simplex = query_tri.simplices[q_idx];
        const auto& c_simplex = cand_tri.simplices[it->second];
        Eigen::Vector2d q_center = (query_pts[q_simplex[0]] + query_pts[q_simplex[1]] +
                                    query_pts[q_simplex[2]]) /
                                   3.0;
        Eigen::Vector2d c_center = (cand_pts[c_simplex[0]] + cand_pts[c_simplex[1]] +
                                    cand_pts[c_simplex[2]]) /
                                   3.0;
        correspondences.emplace_back(q_center, c_center);
    }
    if (correspondences.size() < 2) return result;

    const size_t N = correspondences.size();
    Eigen::MatrixXd query_matrix(N, 2), cand_matrix(N, 2);
    for (size_t i = 0; i < N; ++i) {
        query_matrix.row(i) = correspondences[i].first;
        cand_matrix.row(i) = correspondences[i].second;
    }

    Eigen::Vector2d query_center = query_matrix.colwise().mean();
    Eigen::Vector2d cand_center = cand_matrix.colwise().mean();
    query_matrix.rowwise() -= query_center.transpose();
    cand_matrix.rowwise() -= cand_center.transpose();

    Eigen::Matrix2d H = query_matrix.transpose() * cand_matrix;
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    result.R = svd.matrixV() * svd.matrixU().transpose();
    if (result.R.determinant() < 0.0) {
        Eigen::Matrix2d V = svd.matrixV();
        V.col(1) *= -1.0;
        result.R = V * svd.matrixU().transpose();
    }
    result.t = cand_center - result.R * query_center;

    std::vector<Eigen::Vector2d> transformed_query(query_pts.size());
    for (size_t i = 0; i < query_pts.size(); ++i) {
        transformed_query[i] = result.R * query_pts[i] + result.t;
    }

    Eigen::MatrixXd query_points_matrix(transformed_query.size(), 2);
    for (size_t i = 0; i < transformed_query.size(); ++i) {
        query_points_matrix.row(i) = transformed_query[i];
    }

    using KDTree =
        nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd, 2,
                                            nanoflann::metric_L2_Simple>;
    KDTree query_tree(2, std::ref(query_points_matrix), 10);
    query_tree.index->buildIndex();

    std::vector<TreeMatch> matches;
    for (size_t cand_idx = 0; cand_idx < cand_df.size(); ++cand_idx) {
        Eigen::Vector2d cand_position(
            cand_df[cand_idx].location_x, cand_df[cand_idx].location_y);

        std::vector<size_t> indices(1);
        std::vector<double> d2(1);
        nanoflann::KNNResultSet<double> result_set(1);
        result_set.init(indices.data(), d2.data());
        query_tree.index->findNeighbors(result_set, cand_position.data(), {});

        const double query_radius = GetRadius(query_df[indices[0]]);
        const double cand_radius = GetRadius(cand_df[cand_idx]);
        if (std::abs(query_radius - cand_radius) > 0.4) continue;

        if (std::sqrt(d2[0]) < 0.5) {
            matches.push_back(
                {static_cast<int>(indices[0]), static_cast<int>(cand_idx)});
        }
    }

    if (matches.size() >= 2) {
        const size_t M = matches.size();
        Eigen::MatrixXd refined_query(M, 2), refined_cand(M, 2);
        for (size_t i = 0; i < M; ++i) {
            refined_query.row(i) << query_df[matches[i].query_idx].location_x,
                                     query_df[matches[i].query_idx].location_y;
            refined_cand.row(i) << cand_df[matches[i].cand_idx].location_x,
                                    cand_df[matches[i].cand_idx].location_y;
        }

        query_center = refined_query.colwise().mean();
        cand_center = refined_cand.colwise().mean();
        refined_query.rowwise() -= query_center.transpose();
        refined_cand.rowwise() -= cand_center.transpose();
        H = refined_query.transpose() * refined_cand;

        svd.compute(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        result.R = svd.matrixV() * svd.matrixU().transpose();
        if (result.R.determinant() < 0.0) {
            Eigen::Matrix2d V = svd.matrixV();
            V.col(1) *= -1.0;
            result.R = V * svd.matrixU().transpose();
        }
        result.t = cand_center - result.R * query_center;
    }

    for (size_t i = 0; i < query_pts.size(); ++i) {
        transformed_query[i] = result.R * query_pts[i] + result.t;
        query_points_matrix.row(i) = transformed_query[i];
    }

    KDTree final_query_tree(2, std::ref(query_points_matrix), 10);
    final_query_tree.index->buildIndex();

    matches.clear();
    for (size_t cand_idx = 0; cand_idx < cand_df.size(); ++cand_idx) {
        Eigen::Vector2d cand_position(
            cand_df[cand_idx].location_x, cand_df[cand_idx].location_y);

        std::vector<size_t> indices(1);
        std::vector<double> d2(1);
        nanoflann::KNNResultSet<double> result_set(1);
        result_set.init(indices.data(), d2.data());
        final_query_tree.index->findNeighbors(result_set, cand_position.data(), {});

        if (std::sqrt(d2[0]) < 0.4) {
            matches.push_back(
                {static_cast<int>(indices[0]), static_cast<int>(cand_idx)});
        }
    }

    const int intersection = static_cast<int>(matches.size());
    const int union_count =
        static_cast<int>(query_df.size() + cand_df.size() - intersection);
    result.overlap = union_count > 0 ? static_cast<double>(intersection) / union_count : 0.0;
    result.matches = std::move(matches);
    return result;
}

PrecomputedData ProcessFile(int frame_idx,
                            const DatasetContext& context,
                            const Config& config,
                            const std::vector<RangeBin>& radius_bins) {
    PrecomputedData data;
    const fs::path file =
        context.dataset_root / ("TreeManagerState_" + std::to_string(frame_idx) + ".csv");
    std::ifstream input(file);
    if (!input.good()) {
        return data;
    }

    auto trees = ReadTreeData(file);

    std::vector<Eigen::Matrix4d> scene_axes;
    scene_axes.reserve(trees.size());
    for (const auto& row : trees) {
        if (row.reconstructed != 1) continue;
        if (row.location_x < -30.0 || row.location_x > 30.0 ||
            row.location_y < -30.0 || row.location_y > 30.0) {
            continue;
        }

        Eigen::Matrix4d axis_transform = Eigen::Matrix4d::Identity();
        axis_transform.block<3, 3>(0, 0) = row.R;
        scene_axes.push_back(axis_transform);
    }
    data.scene_transform = ComputeGlobalZAlignmentFallback(scene_axes);

    std::vector<TreeData> aligned_trees;
    aligned_trees.reserve(trees.size());
    for (const auto& row : trees) {
        TreeData transformed = row;
        if (std::isfinite(row.location_z)) {
            transformed = ApplySceneTransform(row, data.scene_transform);
        } else {
            Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
            transform.block<3, 3>(0, 0) = row.R;
            transform.block<3, 1>(0, 3) = Eigen::Vector3d(row.location_x, row.location_y, 0.0);
            transform = data.scene_transform * transform;
            transformed.R = transform.block<3, 3>(0, 0);
            transformed.location_x = transform(0, 3);
            transformed.location_y = transform(1, 3);
            transformed.location_z = row.location_z;
        }
        aligned_trees.push_back(transformed);
    }

    auto selected_trees = SelectTreesForDescriptors(aligned_trees, config.number_of_cluster);
    if (selected_trees.empty()) {
        data.histogram =
            Eigen::MatrixXd::Zero(config.spatial_range_bins.size(), radius_bins.size());
        return data;
    }

    data.histogram =
        ComputeHistogram(selected_trees, config.spatial_range_bins, radius_bins);

    Eigen::MatrixXd kernel(2, 2);
    kernel.setOnes();
    Eigen::MatrixXd hist_smoothed =
        Eigen::MatrixXd::Zero(data.histogram.rows(), data.histogram.cols());

    for (int i = 0; i < data.histogram.rows(); ++i) {
        for (int j = 0; j < data.histogram.cols(); ++j) {
            for (int di = 0; di < 2; ++di) {
                for (int dj = 0; dj < 2; ++dj) {
                    const int ni = i + di;
                    const int nj = j + dj;
                    if (ni < data.histogram.rows() && nj < data.histogram.cols()) {
                        hist_smoothed(i, j) += kernel(di, dj) * data.histogram(ni, nj);
                    }
                }
            }
        }
    }

    data.histogram = hist_smoothed;
    data.centers.resize(selected_trees.size());
    for (size_t i = 0; i < selected_trees.size(); ++i) {
        data.centers[i] =
            Eigen::Vector2d(selected_trees[i].location_x, selected_trees[i].location_y);
    }

    data.tri = ComputeKnnTriangles(
        data.centers, config.knn_k, config.min_dist, config.max_dist);
    data.hash_list = GetTriangleHashes(
        data.tri, data.centers, config.delta_l, config.rho, config.hash_modulus);
    for (const auto& hash : data.hash_list) {
        data.hash_set.insert(hash.first);
    }
    data.df = std::move(selected_trees);
    return data;
}

}  // namespace treeloc
