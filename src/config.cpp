#include "treeloc/config.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace treeloc {

namespace {

std::string Trim(const std::string& text) {
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string StripComment(const std::string& line) {
    bool in_single_quote = false;
    bool in_double_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\'' && !in_double_quote) {
            in_single_quote = !in_single_quote;
            continue;
        }
        if (line[i] == '"' && !in_single_quote) {
            in_double_quote = !in_double_quote;
            continue;
        }
        if (line[i] == '#' && !in_single_quote && !in_double_quote) {
            return line.substr(0, i);
        }
    }
    return line;
}

int CountIndent(const std::string& line) {
    int indent = 0;
    while (indent < static_cast<int>(line.size()) && line[indent] == ' ') {
        ++indent;
    }
    return indent;
}

template <typename NumberType>
NumberType ParseNumber(const std::string& text,
                       const std::string& field_name,
                       size_t line_no) {
    std::stringstream ss(text);
    NumberType value;
    ss >> value;
    if (!ss || !ss.eof()) {
        throw std::runtime_error("Invalid value for '" + field_name + "' on line " +
                                 std::to_string(line_no));
    }
    return value;
}

void AssignScalarField(const std::string& key,
                       const std::string& value,
                       Config& config,
                       size_t line_no) {
    auto parse_string = [](const std::string& text) {
        const std::string trimmed = Trim(text);
        if (trimmed.size() >= 2 &&
            ((trimmed.front() == '"' && trimmed.back() == '"') ||
             (trimmed.front() == '\'' && trimmed.back() == '\''))) {
            return trimmed.substr(1, trimmed.size() - 2);
        }
        return trimmed;
    };

    if (key == "dataset_root") {
        config.dataset_root = parse_string(value);
    } else if (key == "spatial_threshold") {
        config.spatial_threshold = ParseNumber<double>(value, key, line_no);
    } else if (key == "recall_k") {
        config.recall_k = ParseNumber<int>(value, key, line_no);
    } else if (key == "histogram_k") {
        config.histogram_k = ParseNumber<int>(value, key, line_no);
    } else if (key == "knn_k") {
        config.knn_k = ParseNumber<int>(value, key, line_no);
    } else if (key == "min_dist") {
        config.min_dist = ParseNumber<double>(value, key, line_no);
    } else if (key == "max_dist") {
        config.max_dist = ParseNumber<double>(value, key, line_no);
    } else if (key == "delta_l") {
        config.delta_l = ParseNumber<double>(value, key, line_no);
    } else if (key == "rho") {
        config.rho = ParseNumber<long long>(value, key, line_no);
    } else if (key == "hash_modulus") {
        config.hash_modulus = ParseNumber<long long>(value, key, line_no);
    } else if (key == "number_of_cluster") {
        config.number_of_cluster = ParseNumber<int>(value, key, line_no);
    } else if (key == "min_radius") {
        config.min_radius = ParseNumber<double>(value, key, line_no);
    } else if (key == "max_radius") {
        config.max_radius = ParseNumber<double>(value, key, line_no);
    } else if (key == "total_section") {
        config.total_section = ParseNumber<int>(value, key, line_no);
    } else if (key == "bin_width") {
        config.bin_width = ParseNumber<double>(value, key, line_no);
    } else if (key == "spatial_bin_interval") {
        config.spatial_bin_interval = ParseNumber<double>(value, key, line_no);
    } else if (key == "spatial_bin_padding") {
        config.spatial_bin_padding = ParseNumber<double>(value, key, line_no);
    } else if (key == "spatial_bin_count") {
        config.spatial_bin_count = ParseNumber<int>(value, key, line_no);
    } else if (key == "spatial_bin_min") {
        config.spatial_bin_min = ParseNumber<double>(value, key, line_no);
    } else if (key == "spatial_bin_max") {
        config.spatial_bin_max = ParseNumber<double>(value, key, line_no);
    } else {
        throw std::runtime_error("Unknown config key '" + key + "' on line " +
                                 std::to_string(line_no));
    }
}

}  // namespace

std::filesystem::path GetDefaultConfigPath() {
    return std::filesystem::path("config") / "default.yaml";
}

bool LoadConfigFromYaml(const std::filesystem::path& yaml_path,
                        Config& config,
                        std::string* error) {
    try {
        std::ifstream file(yaml_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: " + yaml_path.string());
        }

        std::string line;
        size_t line_no = 0;

        while (std::getline(file, line)) {
            ++line_no;
            const std::string stripped = StripComment(line);
            const std::string trimmed = Trim(stripped);
            if (trimmed.empty()) continue;
            if (CountIndent(line) != 0) {
                throw std::runtime_error("Unsupported indentation on line " +
                                         std::to_string(line_no));
            }

            const size_t colon_pos = trimmed.find(':');
            if (colon_pos == std::string::npos) {
                throw std::runtime_error("Expected 'key: value' on line " +
                                         std::to_string(line_no));
            }

            const std::string key = Trim(trimmed.substr(0, colon_pos));
            const std::string value = Trim(trimmed.substr(colon_pos + 1));
            if (key.empty()) {
                throw std::runtime_error("Empty config key on line " +
                                         std::to_string(line_no));
            }
            AssignScalarField(key, value, config, line_no);
        }

        RefreshDerivedConfig(config);
        if (!ValidateConfig(config, error)) {
            return false;
        }
        return true;
    } catch (const std::exception& ex) {
        if (error != nullptr) {
            *error = ex.what();
        }
        return false;
    }
}

std::vector<RangeBin> BuildSpatialRangeBins(const Config& config) {
    std::vector<RangeBin> bins;
    bins.reserve(config.spatial_bin_count);

    for (int i = 0; i < config.spatial_bin_count; ++i) {
        const double nominal_start = config.spatial_bin_min + i * config.spatial_bin_interval;
        const double nominal_end = nominal_start + config.spatial_bin_interval;
        const double start = std::max(config.spatial_bin_min,
                                      nominal_start - config.spatial_bin_padding);
        const double end = std::min(config.spatial_bin_max,
                                    nominal_end + config.spatial_bin_padding);
        if (end > start) {
            bins.emplace_back(start, end);
        }
    }
    return bins;
}

void RefreshDerivedConfig(Config& config) {
    config.spatial_range_bins = BuildSpatialRangeBins(config);
}

bool ValidateConfig(const Config& config, std::string* error) {
    auto fail = [error](const std::string& message) {
        if (error != nullptr) {
            *error = message;
        }
        return false;
    };

    if (config.dataset_root.empty()) {
        return fail("dataset_root must not be empty");
    }
    if (config.spatial_threshold <= 0.0) {
        return fail("spatial_threshold must be positive");
    }
    if (config.recall_k <= 0) {
        return fail("recall_k must be positive");
    }
    if (config.histogram_k <= 0) {
        return fail("histogram_k must be positive");
    }
    if (config.knn_k <= 0) {
        return fail("knn_k must be positive");
    }
    if (config.min_dist < 0.0 || config.max_dist <= config.min_dist) {
        return fail("max_dist must be greater than min_dist and min_dist must be non-negative");
    }
    if (config.delta_l <= 0.0) {
        return fail("delta_l must be positive");
    }
    if (config.rho <= 0 || config.hash_modulus <= 0) {
        return fail("rho and hash_modulus must be positive");
    }
    if (config.number_of_cluster <= 0) {
        return fail("number_of_cluster must be positive");
    }
    if (config.min_radius < 0.0 || config.max_radius < config.min_radius) {
        return fail("max_radius must be at least min_radius and min_radius must be non-negative");
    }
    if (config.total_section <= 0) {
        return fail("total_section must be positive");
    }
    if (config.bin_width <= 0.0) {
        return fail("bin_width must be positive");
    }
    if (config.spatial_bin_interval <= 0.0) {
        return fail("spatial_bin_interval must be positive");
    }
    if (config.spatial_bin_padding < 0.0) {
        return fail("spatial_bin_padding must be non-negative");
    }
    if (config.spatial_bin_count <= 0) {
        return fail("spatial_bin_count must be positive");
    }
    if (config.spatial_bin_max <= config.spatial_bin_min) {
        return fail("spatial_bin_max must be greater than spatial_bin_min");
    }
    if (config.spatial_bin_interval * config.spatial_bin_count <
        config.spatial_bin_max - config.spatial_bin_min) {
        return fail("spatial_bin_count and spatial_bin_interval must cover spatial_bin_max");
    }

    const std::vector<RangeBin> spatial_range_bins = BuildSpatialRangeBins(config);
    if (spatial_range_bins.empty()) {
        return fail("spatial_range_bins must contain at least one range");
    }

    for (size_t i = 0; i < spatial_range_bins.size(); ++i) {
        const auto& [start, end] = spatial_range_bins[i];
        if (end <= start) {
            return fail("Each spatial_range_bins entry must satisfy end > start");
        }
    }

    return true;
}

}  // namespace treeloc
