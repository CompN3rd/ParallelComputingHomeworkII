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

	children = (TreeElem**)malloc(nx * ny * sizeof(TreeElem**));
	for (int i = 0; i < nx * ny; i++)
	{
		children[i] = new LeafElem();
	}
}

__device__ __host__ BranchElem::~BranchElem()
{
	for (int i = 0; i < nx * ny; i++)
	{
		if (children[i] != NULL)
		{
			delete children[i];
		}
	}

	free(children);
}

__device__ __host__ SWEHandler::SWEHandler(int nx, int ny, int maxRecursions)
{
	this->nx = nx;
	this->ny = ny;

	dim3 block(1);
	dim3 grid(1);

	//create full tree
	createFullTree_Root << <grid, block >> >(&hd, nx, ny, 2);
}

__device__ __host__ SWEHandler::~SWEHandler()
{
	delete hd;
}