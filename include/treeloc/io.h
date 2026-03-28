#pragma once

#include <filesystem>
#include <vector>

#include "treeloc/types.h"

namespace treeloc {

std::vector<Point> ReadTrajectory(const std::filesystem::path& filename);
std::vector<TreeData> ReadTreeData(const std::filesystem::path& filename);

}  // namespace treeloc
