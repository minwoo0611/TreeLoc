<div align="center">

<h1>TreeLoc: 6-DoF LiDAR Global Localization in Forests via Inter-Tree Geometric Matching</h1>

📌 **[ICRA 2026]** Official repository for **TreeLoc**

📄 **[[Paper]](https://arxiv.org/abs/2602.01501)** | 🎥 **[[Video]](https://www.youtube.com/watch?v=1iiqoRSXjUE)**

<a href="https://scholar.google.co.kr/citations?user=aKPTi7gAAAAJ&hl=ko" target="_blank">Minwoo Jung</a>, <a href="https://scholar.google.com/citations?user=ZKuNQGsAAAAJ&hl=en" target="_blank">Nived Chebrolu</a>, <a href="https://scholar.google.com/citations?user=J5pi31QAAAAJ&hl=pt-BR" target="_blank">Lucas Carvalho de Lima</a>, <a href="https://scholar.google.com/citations?user=ISDgG3MAAAAJ&hl=en" target="_blank">Haedam Oh</a>, <a href="https://scholar.google.com/citations?user=BqV8LaoAAAAJ&hl=en" target="_blank">Maurice Fallon</a>, <a href="https://scholar.google.com/citations?user=7yveufgAAAAJ&hl=ko" target="_blank">Ayoung Kim</a><sup>†</sup>

🤝 Collaboration among **Seoul National University** and **University of Oxford**.
<a href="https://www.youtube.com/watch?v=1iiqoRSXjUE" target="_blank">
  <img src="https://github.com/user-attachments/assets/67d137c5-c8a9-4f29-98f9-f32a1fc27056" alt="TreeLoc teaser" width="900" />
</a>

</div>

</div>

### Recent Updates
- [2026/03/28] We uploaded the full TreeLoc code.
- [2026/02/12] Paper released on arXiv
- [2026/02/12] Initial release of the TreeLoc repository

### Contributions
- **TreeLoc is a learning-free global localization framework for forests** that estimates accurate 6-DoF poses using compact tree-level representations.
- **TreeLoc introduces a dual-descriptor pipeline** with TDH for coarse retrieval and a 2D triangle descriptor for fine verification.
- **TreeLoc recovers 6-DoF poses directly from place recognition cues** using only tree geometry, axes, and base heights.
- **TreeLoc achieves strong performance across diverse forest benchmarks** while enabling scalable long-term forest map management.

### Data Source and Tree Extraction

TreeLoc uses tree-level representations extracted from forest LiDAR data. In this repository, the tree observations were prepared using [RealtimeTrees](https://github.com/ori-drs/realtime_trees), and the underlying forest recordings come from the [Oxford Forest Place Recognition Dataset](https://dynamic.robots.ox.ac.uk/datasets/oxford-forest/).

The example dataset bundled with the current configuration is `oxford_single_evo/`, which follows the TreeLoc-ready format described below.

### Prerequisites

TreeLoc has been tested as a C++17 project built with CMake.

- CMake >= 3.16
- A C++17 compiler with OpenMP support
- Eigen3

On Ubuntu, the required packages can be installed with:

```bash
sudo apt update
sudo apt install build-essential cmake libeigen3-dev
```

### Build

```bash
cmake -S . -B build
cmake --build build -j
```

This builds:

- `tree_localization_main`: TreeLoc localization executable

### Input Format

The runtime dataset root should contain:

```text
dataset_root/
├── trajectory.txt
├── TreeManagerState_0.csv
├── TreeManagerState_1.csv
├── TreeManagerState_2.csv
└── ...
```

`trajectory.txt` must contain one pose per line in the format:

```text
timestamp x y z qx qy qz qw
```

The tree-level representations used by TreeLoc were extracted with [RealtimeTrees](https://github.com/ori-drs/realtime_trees). Each `TreeManagerState_<idx>.csv` file is expected to contain tree-level information for the corresponding frame. The required columns are:

- `axis_00` ... `axis_22`
- `location_x`
- `location_y`
- `location_z`
- `dbh` or `dbh_approximation`

The following columns are also supported and used when available:

- `reconstructed`
- `number_clusters`
- `score`

The current parser reads CSV columns by header name, so column order does not need to be fixed as long as the required fields exist.

### Usage

Run TreeLoc with the default configuration:

```bash
./build/tree_localization_main
```

Run TreeLoc with an explicit dataset path and config file:

```bash
./build/tree_localization_main /path/to/dataset config/default.yaml
```

Run TreeLoc by passing only a config file:

```bash
./build/tree_localization_main config/default.yaml
```

### TODO

- [ ] Multi-session support
- [ ] Multi-session dataset release

### Acknowledgement

We thank the Oxford Dynamic Robot Systems Group for releasing the Oxford Forest dataset and the RealtimeTrees project that enabled tree-level data extraction for this work.

If you use this repository, please cite:

```bibtex
@INPROCEEDINGS { mwjung-2026-icra,
    AUTHOR = { Minwoo Jung and Nived Chebrolu and Lucas Carvalho de Lima and Haedam Oh and Maurice Fallon and Ayoung Kim },
    TITLE = { TreeLoc: 6-DoF LiDAR Global Localization in Forests via Inter-Tree Geometric Matching },
    BOOKTITLE = { Proceedings of the IEEE International Conference on Robotics and Automation (ICRA) },
    YEAR = { 2026 },
    MONTH = { June. },
    ADDRESS = { Vienna },
}
```
