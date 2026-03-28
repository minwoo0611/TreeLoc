#include "treeloc/app.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <omp.h>

#include "treeloc/config.h"
#include "treeloc/geometry.h"
#include "treeloc/io.h"
#include "treeloc/matching.h"
#include "treeloc/types.h"

namespace treeloc {

namespace {

struct CandidateInfo {
    int idx;
    double hist_dist;
    double spatial_dist;
    int hash_cnt;
    double overlap;
    double trans_err;
    Eigen::Matrix2d R;
    Eigen::Vector2d t;
    std::vector<TreeMatch> matches;
};

void PrintSummary(int max_idx,
                  const Config& config,
                  double recall_at_100_precision,
                  int TP,
                  int SP_3dof,
                  int SP2_3dof,
                  int SP_6dof,
                  int SP2_6dof,
                  int N,
                  double total_translation_error,
                  double total_z_error,
                  double total_yaw_error,
                  double total_roll_error,
                  double total_pitch_error,
                  double total_translation_error_6dof,
                  double total_rot_error,
                  double max_f1,
                  double auc_score,
                  double matching_time,
                  const std::chrono::high_resolution_clock::time_point& start_time,
                  const std::chrono::high_resolution_clock::time_point& end_time) {
    const double recall_at_k = N > 0 ? static_cast<double>(TP) / N : 0.0;
    const double success_at_k = N > 0 ? static_cast<double>(SP_3dof) / N : 0.0;
    const double success_at_k2 = TP > 0 ? static_cast<double>(SP2_3dof) / TP : 0.0;
    const double average_translation_error =
        SP2_3dof > 0 ? total_translation_error / SP2_3dof : 0.0;
    const double average_z_error = SP2_3dof > 0 ? total_z_error / SP2_3dof : 0.0;
    const double average_yaw_error = SP2_3dof > 0 ? total_yaw_error / SP2_3dof : 0.0;
    const double average_roll_error = SP2_3dof > 0 ? total_roll_error / SP2_3dof : 0.0;
    const double average_pitch_error =
        SP2_3dof > 0 ? total_pitch_error / SP2_3dof : 0.0;

    std::cout << "Recall@100% Precision: " << recall_at_100_precision << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average Translation Error: " << average_translation_error << "m\n";
    std::cout << "Average Z Error: " << average_z_error << "m\n";
    std::cout << "Average Yaw Error: " << average_yaw_error / M_PI * 180.0 << " degrees\n";
    std::cout << "Average Roll Error: " << average_roll_error / M_PI * 180.0 << " degrees\n";
    std::cout << "Average Pitch Error: " << average_pitch_error / M_PI * 180.0
              << " degrees\n";
    std::cout << "Max F1 Score: " << max_f1 << "\n";
    std::cout << "AUC Score: " << auc_score << "\n";
    std::cout << "Recall@" << config.recall_k << ": " << recall_at_k << "\n";
    std::cout << "Success_3dof@" << config.recall_k << ": " << success_at_k << "\n";
    std::cout << "Success2_3dof@" << config.recall_k << ": " << success_at_k2 << "\n";
    std::cout << "Success_6dof: " << (N > 0 ? static_cast<double>(SP_6dof) / N : 0.0)
              << "\n";
    std::cout << "Success2_6dof: "
              << (TP > 0 ? static_cast<double>(SP2_6dof) / TP : 0.0) << "\n";
    std::cout << "Average Translation Error (6dof): "
              << (SP2_6dof > 0 ? total_translation_error_6dof / SP2_6dof : 0.0)
              << "m\n";
    std::cout << "Average Rotation Error (6dof): "
              << (SP2_6dof > 0 ? total_rot_error / M_PI * 180.0 / SP2_6dof : 0.0)
              << " degrees\n";
    std::cout << "True Positives: " << TP << "\n";
    std::cout << "Total Evaluated: " << N << "\n";
    std::cout << "Number of cases with valid candidates (≤" << config.spatial_threshold
              << "m): " << N << "\n";
    std::cout << "Percentage of cases with valid candidates: "
              << (N / static_cast<double>(max_idx + 1) * 100.0) << "%\n";
    std::cout << "Descriptor construction time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
                         .count() /
                     static_cast<double>(max_idx + 1)
              << " ms per frame.\n";
    std::cout << "Total matching time per frame: " << matching_time / N << " seconds.\n";
}

}  // namespace

int RunLocalizationCli(int argc, char** argv) {
    if (argc > 3) {
        std::cerr << "Usage: " << argv[0] << " [dataset_root|config_yaml] [config_yaml]\n";
        return 1;
    }

    Config config;
    std::string config_error;
    fs::path config_path = GetDefaultConfigPath();
    fs::path dataset_override;
    bool has_dataset_override = false;
    bool config_loaded = false;

    auto is_yaml_path = [](const fs::path& path) {
        const std::string extension = path.extension().string();
        return extension == ".yaml" || extension == ".yml";
    };

    if (argc == 2) {
        const fs::path arg1 = argv[1];
        if (is_yaml_path(arg1)) {
            config_path = arg1;
        } else {
            dataset_override = arg1;
            has_dataset_override = true;
        }
    } else if (argc == 3) {
        dataset_override = argv[1];
        has_dataset_override = true;
        config_path = argv[2];
    }

    if (fs::exists(config_path)) {
        if (!LoadConfigFromYaml(config_path, config, &config_error)) {
            std::cerr << config_error << "\n";
            return 1;
        }
        config_loaded = true;
    } else if (argc >= 2 && is_yaml_path(config_path)) {
        std::cerr << "Could not open config file: " << config_path << "\n";
        return 1;
    }

    if (has_dataset_override) {
        config.dataset_root = dataset_override;
    } else if (config_loaded && config.dataset_root.is_relative()) {
        const fs::path config_relative_dataset_root =
            config_path.parent_path() / config.dataset_root;
        if (fs::exists(config_relative_dataset_root)) {
            config.dataset_root = config_relative_dataset_root;
        }
    }

    if (config.dataset_root.empty()) {
        std::cerr << "dataset_root must be provided through CLI or YAML\n";
        return 1;
    }

    RefreshDerivedConfig(config);
    if (!ValidateConfig(config, &config_error)) {
        std::cerr << config_error << "\n";
        return 1;
    }

    const fs::path dataset_root = fs::absolute(config.dataset_root);
    if (!fs::exists(dataset_root)) {
        std::cerr << "dataset_root does not exist: " << dataset_root << "\n";
        return 1;
    }

    const std::vector<RangeBin> radius_bins = BuildRadiusBins(config);

    DatasetContext context;
    context.dataset_root = dataset_root;
    context.trajectory_full = ReadTrajectory(context.dataset_root / "trajectory.txt");
    if (context.trajectory_full.empty()) {
        std::cerr << "trajectory.txt not found, exiting.\n";
        return 1;
    }

    const int max_idx = static_cast<int>(context.trajectory_full.size()) - 1;
    double matching_time = 0.0;

    std::vector<PrecomputedData> precomputed(max_idx + 1);
    std::vector<int> valid_indices;
    std::vector<Eigen::MatrixXd> histograms(max_idx + 1);

    const auto start_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i <= max_idx; ++i) {
        auto data = ProcessFile(i, context, config, radius_bins);
        if (!data.df.empty()) {
            #pragma omp critical
            {
                histograms[i] = data.histogram;
                precomputed[i] = data;
                valid_indices.push_back(i);
            }
        }
    }
    const auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Descriptor construction time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
                     .count()
              << " ms" << std::endl;

    std::sort(valid_indices.begin(), valid_indices.end());

    std::vector<Eigen::Vector2d> kd_points;
    kd_points.reserve(valid_indices.size());
    for (int index : valid_indices) {
        kd_points.emplace_back(
            context.trajectory_full[index].x, context.trajectory_full[index].y);
    }

    Eigen::MatrixXd kd_matrix(kd_points.size(), 2);
    for (size_t i = 0; i < kd_points.size(); ++i) {
        kd_matrix.row(i) = kd_points[i];
    }

    using KDTree =
        nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd, 2,
                                            nanoflann::metric_L2_Simple>;
    KDTree kdtree(2, std::ref(kd_matrix), 10);
    kdtree.index->buildIndex();

    std::vector<double> thresholds(1000);
    for (size_t i = 0; i < thresholds.size(); ++i) {
        thresholds[i] = i / 999.0;
    }

    std::vector<double> tp(thresholds.size(), 0.0);
    std::vector<double> fp(thresholds.size(), 0.0);
    std::vector<double> tn(thresholds.size(), 0.0);
    std::vector<double> fn(thresholds.size(), 0.0);

    int TP = 0;
    int SP_3dof = 0;
    int SP2_3dof = 0;
    int N = 0;
    int SP_6dof = 0;
    int SP2_6dof = 0;
    double total_translation_error = 0.0;
    double total_yaw_error = 0.0;
    double total_translation_error_6dof = 0.0;
    double total_rot_error = 0.0;
    double total_z_error = 0.0;
    double total_roll_error = 0.0;
    double total_pitch_error = 0.0;

    for (int current_idx = 0; current_idx <= max_idx; ++current_idx) {
        if (current_idx >= static_cast<int>(histograms.size()) ||
            current_idx >= static_cast<int>(precomputed.size())) {
            continue;
        }

        const auto& hist_current = histograms.at(current_idx);

        std::vector<int> candidates;
        for (int idx = 0; idx < static_cast<int>(histograms.size()); ++idx) {
            if (idx <= current_idx &&
                std::abs(idx - current_idx) > 50 &&
                idx < static_cast<int>(precomputed.size())) {
                candidates.push_back(idx);
            }
        }
        if (candidates.empty()) continue;

        Eigen::Vector2d query_point(
            context.trajectory_full[current_idx].x,
            context.trajectory_full[current_idx].y);

        std::vector<double> d2(kd_points.size());
        std::vector<size_t> kd_indices(kd_points.size());
        nanoflann::KNNResultSet<double> result_set(kd_points.size());
        result_set.init(kd_indices.data(), d2.data());
        kdtree.index->findNeighbors(result_set, query_point.data(), {});

        std::vector<int> gt_list;
        for (size_t i = 0; i < kd_points.size(); ++i) {
            if (std::sqrt(d2[i]) > config.spatial_threshold) break;
            const int original_idx = valid_indices[kd_indices[i]];
            if (std::binary_search(candidates.begin(), candidates.end(), original_idx)) {
                gt_list.push_back(original_idx);
            }
        }
        if (gt_list.empty()) continue;
        ++N;

        const auto matching_time_start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<int, double>> distances_hist;
        distances_hist.reserve(candidates.size());

        #pragma omp parallel
        {
            std::vector<std::pair<int, double>> local_distances;
            #pragma omp for nowait
            for (int cand_idx : candidates) {
                double dist = std::numeric_limits<double>::infinity();
                if (histograms.at(cand_idx).rows() != 0) {
                    dist = ChiSquaredDistance(hist_current, histograms.at(cand_idx));
                }
                local_distances.emplace_back(cand_idx, dist);
            }
            #pragma omp critical
            distances_hist.insert(
                distances_hist.end(), local_distances.begin(), local_distances.end());
        }
        if (distances_hist.empty()) continue;

        std::sort(distances_hist.begin(), distances_hist.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
        const int hist_k =
            std::min(config.histogram_k, static_cast<int>(distances_hist.size()));
        std::vector<std::pair<int, double>> top_hist(
            distances_hist.begin(), distances_hist.begin() + hist_k);

        std::vector<CandidateInfo> infos;
        infos.reserve(top_hist.size());
        for (const auto& [cand_idx, hist_dist] : top_hist) {
            const auto& query_hashes = precomputed[current_idx].hash_list;
            const auto& cand_hashes = precomputed[cand_idx].hash_list;
            std::unordered_set<long long> cand_hash_set;
            for (const auto& [hash, _] : cand_hashes) {
                cand_hash_set.insert(hash);
            }

            int common_hashes = 0;
            for (const auto& [hash, _] : query_hashes) {
                if (cand_hash_set.count(hash)) ++common_hashes;
            }

            const double spatial_dist =
                (query_point - Eigen::Vector2d(
                                   context.trajectory_full[cand_idx].x,
                                   context.trajectory_full[cand_idx].y))
                    .norm();

            infos.push_back({cand_idx,
                             hist_dist,
                             spatial_dist,
                             common_hashes,
                             0.0,
                             std::numeric_limits<double>::infinity(),
                             Eigen::Matrix2d::Identity(),
                             Eigen::Vector2d::Zero(),
                             {}});
        }

        std::sort(infos.begin(), infos.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.hash_cnt > rhs.hash_cnt; });
        if (infos.size() > 10) infos.resize(10);

        #pragma omp parallel for
        for (size_t i = 0; i < infos.size(); ++i) {
            auto& info = infos[i];
            auto result = Compute2DTransformation(
                precomputed[current_idx].centers,
                precomputed[info.idx].centers,
                precomputed[current_idx].tri,
                precomputed[info.idx].tri,
                precomputed[current_idx].hash_list,
                precomputed[info.idx].hash_list,
                precomputed[current_idx].df,
                precomputed[info.idx].df);

            info.overlap = result.overlap;
            info.trans_err = result.t.norm();
            info.R = result.R;
            info.t = result.t;
            info.matches = std::move(result.matches);
        }

        int best_idx = -1;
        double best_overlap = -1.0;
        Eigen::Matrix2d best_R = Eigen::Matrix2d::Identity();
        Eigen::Vector2d best_t = Eigen::Vector2d::Zero();
        std::vector<TreeMatch> best_matches;

        for (const auto& info : infos) {
            if (best_overlap <= info.overlap) {
                best_idx = info.idx;
                best_overlap = info.overlap;
                best_R = info.R;
                best_t = info.t;
                best_matches = info.matches;
            }
        }

        const auto matching_time_end = std::chrono::high_resolution_clock::now();
        matching_time +=
            std::chrono::duration<double>(matching_time_end - matching_time_start).count();

        double spatial_dist = 100000.0;
        if (best_idx != -1) {
            Eigen::Vector2d candidate_position(
                context.trajectory_full[best_idx].x,
                context.trajectory_full[best_idx].y);
            spatial_dist = (query_point - candidate_position).norm();

            Eigen::Vector3d current_position(
                context.trajectory_full[current_idx].x,
                context.trajectory_full[current_idx].y,
                context.trajectory_full[current_idx].z);
            Eigen::Quaterniond current_quaternion(
                context.trajectory_full[current_idx].qw,
                context.trajectory_full[current_idx].qx,
                context.trajectory_full[current_idx].qy,
                context.trajectory_full[current_idx].qz);

            Eigen::Vector3d best_position(
                context.trajectory_full[best_idx].x,
                context.trajectory_full[best_idx].y,
                context.trajectory_full[best_idx].z);
            Eigen::Quaterniond best_quaternion(
                context.trajectory_full[best_idx].qw,
                context.trajectory_full[best_idx].qx,
                context.trajectory_full[best_idx].qy,
                context.trajectory_full[best_idx].qz);

            Eigen::Matrix3d R_current = current_quaternion.toRotationMatrix();
            Eigen::Matrix3d R_best = best_quaternion.toRotationMatrix();

            Eigen::Vector3d pos_rel_gt =
                R_current.transpose() * (best_position - current_position);
            Eigen::Matrix3d R_rel_gt = R_current.transpose() * R_best;

            double gt_yaw;
            double gt_pitch;
            double gt_roll;
            EulerZYX(R_rel_gt, gt_yaw, gt_pitch, gt_roll);

            Eigen::Matrix2d R_pred2d = best_R.transpose();
            Eigen::Vector2d t_pred2d = -R_pred2d * best_t;

            Eigen::Matrix3d R_pred3 = Eigen::Matrix3d::Identity();
            R_pred3.block<2, 2>(0, 0) = R_pred2d;
            Eigen::Vector3d t_pred3(t_pred2d(0), t_pred2d(1), 0.0);

            const Eigen::Matrix4d T_current = precomputed[current_idx].scene_transform;
            const Eigen::Matrix4d T_best = precomputed[best_idx].scene_transform;

            std::vector<double> query_heights;
            std::vector<double> cand_heights;
            query_heights.reserve(best_matches.size());
            cand_heights.reserve(best_matches.size());

            for (const auto& match : best_matches) {
                if (match.query_idx < 0 || match.cand_idx < 0) continue;
                if (match.query_idx >= static_cast<int>(precomputed[current_idx].df.size()) ||
                    match.cand_idx >= static_cast<int>(precomputed[best_idx].df.size())) {
                    continue;
                }

                const auto& query_tree = precomputed[current_idx].df[match.query_idx];
                const auto& cand_tree = precomputed[best_idx].df[match.cand_idx];
                if (!std::isfinite(query_tree.location_z) ||
                    !std::isfinite(cand_tree.location_z)) {
                    continue;
                }

                query_heights.push_back(query_tree.location_z);
                cand_heights.push_back(cand_tree.location_z);
            }

            if (!query_heights.empty()) {
                double z_offset = 0.0;
                if (query_heights.size() == 1) {
                    z_offset = query_heights[0] - cand_heights[0];
                } else {
                    z_offset = RansacZOffset(
                        query_heights,
                        cand_heights,
                        200,
                        std::min<size_t>(10, query_heights.size()),
                        0.1,
                        0.9);
                }
                t_pred3(2) += z_offset;
            }

            Eigen::Matrix4d T_pred = Eigen::Matrix4d::Identity();
            T_pred.block<3, 3>(0, 0) = R_pred3;
            T_pred.block<3, 1>(0, 3) = t_pred3;

            const Eigen::Matrix4d T_rel_pred = T_current.inverse() * T_pred * T_best;
            const Eigen::Matrix3d R_rel_pred = T_rel_pred.block<3, 3>(0, 0);
            const Eigen::Vector3d pos_rel_pred = T_rel_pred.block<3, 1>(0, 3);

            double pred_yaw;
            double pred_pitch;
            double pred_roll;
            EulerZYX(R_rel_pred, pred_yaw, pred_pitch, pred_roll);

            const double trans_err =
                (pos_rel_gt.head<2>() - pos_rel_pred.head<2>()).norm();
            const double trans_err_3d = (pos_rel_gt - pos_rel_pred).norm();

            const double yaw_err = std::abs(WrapAngle(gt_yaw - pred_yaw));
            const double pitch_err = std::abs(WrapAngle(gt_pitch - pred_pitch));
            const double roll_err = std::abs(WrapAngle(gt_roll - pred_roll));
            const double rot_err =
                std::sqrt(yaw_err * yaw_err + pitch_err * pitch_err + roll_err * roll_err);

            const double z_err = std::abs(pos_rel_pred(2) - pos_rel_gt(2));

            if (spatial_dist <= config.spatial_threshold) {
                ++TP;

                if (trans_err <= 0.5 && std::abs(yaw_err) * 180.0 / M_PI <= 5.0) {
                    ++SP2_3dof;
                    total_translation_error += trans_err;
                    total_z_error += z_err;
                    total_yaw_error += yaw_err;
                    total_roll_error += roll_err;
                    total_pitch_error += pitch_err;
                }

                if (trans_err_3d <= 0.5 && std::abs(rot_err) * 180.0 / M_PI <= 5.0) {
                    ++SP2_6dof;
                    total_translation_error_6dof += trans_err_3d;
                    total_rot_error += rot_err;
                }
            }

            if (trans_err <= 0.5 && std::abs(yaw_err) * 180.0 / M_PI <= 5.0) {
                ++SP_3dof;
            }

            if (trans_err_3d <= 0.5 && std::abs(rot_err) * 180.0 / M_PI <= 5.0) {
                ++SP_6dof;
            }
        }

        const bool is_true_neighbor = (spatial_dist <= config.spatial_threshold);
        for (size_t t = 0; t < thresholds.size(); ++t) {
            if (best_overlap >= thresholds[t]) {
                (is_true_neighbor ? tp : fp)[t]++;
            } else {
                (is_true_neighbor ? fn : tn)[t]++;
            }
        }
    }

    std::vector<double> precisions(thresholds.size());
    std::vector<double> recalls(thresholds.size());
    std::vector<double> f1_scores(thresholds.size());
    for (size_t i = 0; i < thresholds.size(); ++i) {
        precisions[i] = (tp[i] + fp[i] > 0.0) ? tp[i] / (tp[i] + fp[i]) : 0.0;
        recalls[i] = (tp[i] + fn[i] > 0.0) ? tp[i] / (tp[i] + fn[i]) : 0.0;
        f1_scores[i] =
            (precisions[i] + recalls[i] > 0.0)
                ? 2.0 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])
                : 0.0;
    }

    const double max_f1 = *std::max_element(f1_scores.begin(), f1_scores.end());
    double recall_at_100_precision = 0.0;
    for (size_t i = 0; i < thresholds.size(); ++i) {
        if (precisions[i] >= 1.0) {
            recall_at_100_precision = recalls[i];
            break;
        }
    }
    std::vector<size_t> sort_indices(recalls.size());
    std::iota(sort_indices.begin(), sort_indices.end(), 0);
    std::sort(sort_indices.begin(), sort_indices.end(),
              [&recalls](size_t lhs, size_t rhs) { return recalls[lhs] < recalls[rhs]; });

    double auc_score = 0.0;
    if (sort_indices.size() > 1) {
        double prev_recall = recalls[sort_indices[0]];
        for (size_t i = 1; i < sort_indices.size(); ++i) {
            auc_score +=
                precisions[sort_indices[i]] * (recalls[sort_indices[i]] - prev_recall);
            prev_recall = recalls[sort_indices[i]];
        }
    }

    PrintSummary(max_idx,
                 config,
                 recall_at_100_precision,
                 TP,
                 SP_3dof,
                 SP2_3dof,
                 SP_6dof,
                 SP2_6dof,
                 N,
                 total_translation_error,
                 total_z_error,
                 total_yaw_error,
                 total_roll_error,
                 total_pitch_error,
                 total_translation_error_6dof,
                 total_rot_error,
                 max_f1,
                 auc_score,
                 matching_time,
                 start_time,
                 end_time);
    return 0;
}

}  // namespace treeloc
