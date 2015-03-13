#pragma once

#include "treeSolver.h"

__device__ __host__ void createFullTree(TreeElem** root, int nx, int ny, int base);
__device__ __host__ int countLeaves(TreeElem* root);
__global__ void deleteChildren(TreeElem** children);