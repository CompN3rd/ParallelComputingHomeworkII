#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//gpu tree
class TreeElem 
{
public:
	virtual __device__ __host__ bool isLeaf() = 0;
};

class LeafElem : public TreeElem
{
public:
	float value;

	__device__ __host__ LeafElem();

	virtual __device__ __host__ bool isLeaf()
	{
		return true;
	}
};

class BranchElem : TreeElem
{
public:
	int nx;
	int ny;
	TreeElem** children;

	__device__ __host__ BranchElem(int nx, int ny);
	__device__ __host__ ~BranchElem();

	virtual __device__ __host__ bool isLeaf()
	{
		return false;
	}
};

//gpu handler
class SWEHandler
{
public:
	//Coarsest grids for 
	TreeElem** hd;
	TreeElem** hud;
	TreeElem** hvd;

	TreeElem** Fhd;
	TreeElem** Fhud;
	TreeElem** Fhvd;
	TreeElem** Ghd;
	TreeElem** Ghud;
	TreeElem** Ghvd;

	// arrays to hold the bathymetry source terms for the hu and hv equations
	TreeElem** Bxd;
	TreeElem** Byd;

	// helper arrays: store maximum height and velocities to determine time step
	float* maxhd;
	float* maxhud;
	float* maxhvd;

	__device__ __host__ SWEHandler(int nx, int ny, int maxRecursions);
};