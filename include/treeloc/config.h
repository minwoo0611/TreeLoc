#pragma once

#include <filesystem>
#include <utility>
#include <vector>
#include <string>

namespace treeloc {

using RangeBin = std::pair<double, double>;

struct Config {
    std::filesystem::path dataset_root = "oxford_single_evo";
    double spatial_threshold = 10.0;
    int recall_k = 1;
    int histogram_k = 100;
    int knn_k = 10;
    double min_dist = 2.0;
    double max_dist = 30.0;
    double delta_l = 0.1;
    long long rho = 100007;
    long long hash_modulus = 50000000;
    int number_of_cluster = 30;
    double min_radius = 0.0;
    double max_radius = 0.8;
    int total_section = 8;
    double bin_width = 0.15;
    double spatial_bin_interval = 6.0;
    double spatial_bin_padding = 1.0;
    int spatial_bin_count = 5;
    double spatial_bin_min = 0.0;
    double spatial_bin_max = 30.0;
    std::vector<RangeBin> spatial_range_bins = {
        {0.0, 7.0}, {5.0, 13.0}, {11.0, 19.0}, {17.0, 25.0}, {23.0, 30.0}
    };
};

std::filesystem::path GetDefaultConfigPath();
bool LoadConfigFromYaml(const std::filesystem::path& yaml_path,
                        Config& config,
                        std::string* error = nullptr);
std::vector<RangeBin> BuildSpatialRangeBins(const Config& config);
void RefreshDerivedConfig(Config& config);
bool ValidateConfig(const Config& config, std::string* error = nullptr);
std::vector<RangeBin> BuildRadiusBins(const Config& config);

}  // namespace treeloc
