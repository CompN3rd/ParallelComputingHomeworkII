#include "treeSolver.h"
#include "treeOperations.h"

__device__ __host__ LeafElem::LeafElem()
{
	value = 0.0f;
}

__device__ __host__ BranchElem::BranchElem(int nx, int ny, int depth)
{
	this->nx = nx;
	this->ny = ny;
	this->depth = depth;

	children = new TreeElem*[nx * ny];
	//for (int i = 0; i < nx * ny; i++)
	//{
	//	children[i] = NULL;
	//}
}

__device__ __host__ BranchElem::~BranchElem()
{
	dim3 grid(1);
	dim3 block(nx * ny);
	deleteChildren << <grid, block >> >(children);

	delete[] children;
}

__device__ __host__ SWEHandler::SWEHandler(int nx, int ny, int maxRecursions)
{
	this->nx = nx;
	this->ny = ny;

	//create full tree
	createFullTree(&hd, nx, ny, 2);
}

__device__ __host__ SWEHandler::~SWEHandler()
{
	delete hd;
}