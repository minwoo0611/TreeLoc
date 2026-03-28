#include "treeloc/io.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace treeloc {

namespace {

std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) {
        cells.push_back(cell);
    }
    if (!line.empty() && line.back() == ',') {
        cells.emplace_back();
    }
    return cells;
}

bool TryGetCell(const std::vector<std::string>& cells,
                const std::unordered_map<std::string, size_t>& columns,
                const std::string& key,
                std::string& value) {
    auto it = columns.find(key);
    if (it == columns.end() || it->second >= cells.size()) {
        return false;
    }
    value = cells[it->second];
    return true;
}

double GetRequiredDouble(const std::vector<std::string>& cells,
                         const std::unordered_map<std::string, size_t>& columns,
                         const std::string& key) {
    std::string value;
    if (!TryGetCell(cells, columns, key, value) || value.empty()) {
        throw std::invalid_argument("Empty " + key);
    }
    return std::stod(value);
}

double GetOptionalDouble(const std::vector<std::string>& cells,
                         const std::unordered_map<std::string, size_t>& columns,
                         const std::string& key,
                         double fallback) {
    std::string value;
    if (!TryGetCell(cells, columns, key, value) || value.empty()) {
        return fallback;
    }
    return std::stod(value);
}

}  // namespace

std::vector<Point> ReadTrajectory(const std::filesystem::path& filename) {
    std::vector<Point> trajectory;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        Point point;
        double time;
        ss >> time >> point.x >> point.y >> point.z
           >> point.qx >> point.qy >> point.qz >> point.qw;
        trajectory.push_back(point);
    }
    return trajectory;
}

std::vector<TreeData> ReadTreeData(const std::filesystem::path& filename) {
    std::vector<TreeData> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening CSV file: " << filename << '\n';
        return data;
    }

    std::string line;
    if (!std::getline(file, line)) {
        return data;
    }

    auto header = SplitCsvLine(line);
    std::unordered_map<std::string, size_t> columns;
    for (size_t i = 0; i < header.size(); ++i) {
        columns[header[i]] = i;
    }

    int row_num = 0;
    while (std::getline(file, line)) {
        ++row_num;
        try {
            TreeData td;
            auto cells = SplitCsvLine(line);

            td.R(0, 0) = GetRequiredDouble(cells, columns, "axis_00");
            td.R(0, 1) = GetRequiredDouble(cells, columns, "axis_01");
            td.R(0, 2) = GetRequiredDouble(cells, columns, "axis_02");
            td.R(1, 0) = GetRequiredDouble(cells, columns, "axis_10");
            td.R(1, 1) = GetRequiredDouble(cells, columns, "axis_11");
            td.R(1, 2) = GetRequiredDouble(cells, columns, "axis_12");
            td.R(2, 0) = GetRequiredDouble(cells, columns, "axis_20");
            td.R(2, 1) = GetRequiredDouble(cells, columns, "axis_21");
            td.R(2, 2) = GetRequiredDouble(cells, columns, "axis_22");
            td.location_x = GetRequiredDouble(cells, columns, "location_x");
            td.location_y = GetRequiredDouble(cells, columns, "location_y");
            td.location_z = GetOptionalDouble(
                cells, columns, "location_z", std::numeric_limits<double>::quiet_NaN());
            td.dbh = GetOptionalDouble(
                cells, columns, "dbh", std::numeric_limits<double>::quiet_NaN());
            td.dbh_approximation = GetOptionalDouble(
                cells, columns, "dbh_approximation", std::numeric_limits<double>::quiet_NaN());
            if (!std::isfinite(td.dbh) && !std::isfinite(td.dbh_approximation)) {
                throw std::invalid_argument("Either dbh or dbh_approximation is required");
            }
            if (!std::isfinite(td.dbh)) {
                td.dbh = td.dbh_approximation;
            }
            if (!std::isfinite(td.dbh_approximation)) {
                td.dbh_approximation = td.dbh;
            }

            td.score = GetOptionalDouble(cells, columns, "score", 1.0);

            std::string reconstructed;
            if (!TryGetCell(cells, columns, "reconstructed", reconstructed) ||
                reconstructed.empty()) {
                td.reconstructed = 1;
            } else if (reconstructed == "True" || reconstructed == "true" ||
                       reconstructed == "1") {
                td.reconstructed = 1;
            } else if (reconstructed == "False" || reconstructed == "false" ||
                       reconstructed == "0") {
                td.reconstructed = 0;
            } else {
                throw std::invalid_argument("Invalid reconstructed value: " + reconstructed);
            }

            std::string number_clusters;
            if (!TryGetCell(cells, columns, "number_clusters", number_clusters) ||
                number_clusters.empty()) {
                td.number_clusters = 3;
            } else {
                td.number_clusters = std::stoi(number_clusters);
            }

            if (td.location_x < -30.0 || td.location_x > 30.0 ||
                td.location_y < -30.0 || td.location_y > 30.0) {
                continue;
            }

            data.push_back(td);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing row " << row_num << " in " << filename
                      << ": " << e.what() << " (row content: " << line << ")\n";
        }
    }

    return data;
}

}  // namespace treeloc
