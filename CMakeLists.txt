cmake_minimum_required(VERSION 3.16)

project(TreeLoc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_library(treeloc
    src/config.cpp
    src/io.cpp
    src/geometry.cpp
    src/matching.cpp
    src/app.cpp
)

target_include_directories(treeloc
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(treeloc
    PUBLIC
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
)

add_executable(tree_localization_main tree_localization_main.cpp)
target_link_libraries(tree_localization_main PRIVATE treeloc)
