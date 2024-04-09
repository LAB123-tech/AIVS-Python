# AIVS (Approximate Intrinsic Voxel Structure) Python Implementation

This is a Python implementation of the Approximate Intrinsic Voxel Structure (AIVS) algorithm proposed in the paper titled "Approximate Intrinsic Voxel Structure for Point Cloud Simplification", published in TIP-2021. The original code implementation can be found at https://github.com/vvvwo/AIVS-project.

## Introduction

AIVS is a method for simplifying point clouds by approximating their intrinsic voxel structure. This implementation supports batch computation based on AIVS, with the entire process divided into the following 4 steps:

1. Preprocessing: Voxelization of point clouds
2. Grouping and parallel sampling of the first set of point clouds
3. Grouping and parallel sampling of the second set of point clouds
4. Post-processing: Removal of redundant points

## Worth-note

- It's worth noting that in the original paper, octree downsampling is used first to uniformize the point cloud. Here, for convenience, I directly use farthest point sampling to accomplish this.

- Due to limitations in the Python language, the sampling process for individual point clouds has been minimized as much as possible. 
- AIVS sampling on a point cloud with 2600 points now takes less than 0.5 seconds. Feel free to modify it to CUDA code for even faster implementation.
- Since knn_cuda cannot be installed on Windows, it's recommended to use this program on Ubuntu for optimal performance.

## Installation

To install AIVS, clone this repository and install the required dependencies using pip:

```bash
git clone https://github.com/your_username/AIVS-python.git
cd AIVS-python
pip install -r requirements.txt
```

## Results

As the number of points decreases, the results sampled by AIVS retain the essential structural features of the original point cloud remarkably well. This superiority over FPS highlights its effectiveness.

<p align="center">
  <img src="AIVS-Python/pic/demo.png">
</p>
