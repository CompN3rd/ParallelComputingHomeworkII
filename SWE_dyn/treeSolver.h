#pragma once

#include <cstdlib>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <helper_cuda.h>

//gpu tree
class TreeElem 
{
public:
	virtual __device__ __host__ bool isLeaf() = 0;
	virtual __device__ __host__ ~TreeElem(){};
};

class LeafElem : public TreeElem
{
public:
	float value;

	__device__ __host__ LeafElem();
	virtual __device__ __host__ ~LeafElem(){};

	virtual __device__ __host__ bool isLeaf()
	{
		return true;
	}
};

class BranchElem : public TreeElem
{
public:
	int nx;
	int ny;
	int depth;
	TreeElem** children;

	__device__ __host__ BranchElem(int nx, int ny, int depth);
	virtual __device__ __host__ ~BranchElem();

	virtual __device__ __host__ bool isLeaf()
	{
		return false;
	}
};

//gpu handler
class SWEHandler
{
public:
	//resolution of full grid
	int nx;
	int ny;

	//Coarsest grids for 
	TreeElem* hd;
	TreeElem* hud;
	TreeElem* hvd;

	TreeElem* Fhd;
	TreeElem* Fhud;
	TreeElem* Fhvd;
	TreeElem* Ghd;
	TreeElem* Ghud;
	TreeElem* Ghvd;

	// arrays to hold the bathymetry source terms for the hu and hv equations
	TreeElem* Bxd;
	TreeElem* Byd;

	__device__ __host__ SWEHandler(int nx, int ny, int maxRecursions);
	__device__ __host__ ~SWEHandler();
};