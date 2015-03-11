#include "treeSolver.h"

__device__ __host__ LeafElem::LeafElem()
{
	value = 0.0f;
}

__device__ __host__ BranchElem::BranchElem(int nx, int ny)
{
	this->nx = nx;
	this->ny = ny;

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

__device__ __host__ SWEHandler::SWEHandler(int nx, int ny)
{
}

__global__ void setSon(TreeElem** grid, int nx, int ny, float val)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= nx || j >= ny)
		return;

	int idx = j * nx + i;

	if (grid[idx]->isLeaf())
	{
		((LeafElem*)grid[idx])->value = val;
	}
}

//masterprozess
__global__ void setFather(float* values, int nx, int ny)
{
	dim3 block(nx, ny);
	dim3 grid(nx, ny);

	//tree
	BranchElem* hd = new BranchElem(16, 16);

	for (int i = 0; i < nx * ny; i++)
	{
		if (hd->children[i]->isLeaf())
		{
			((LeafElem*)hd->children[i])->value = values[i];
		}
	}
	
	setSon << <grid, block >> >(hd->children, 16, 16, 1.0f);

	for (int i = 0; i < nx * ny; i++)
	{
		if (hd->children[i]->isLeaf())
		{
			values[i] = ((LeafElem*)hd->children[i])->value;
		}
	}

	delete hd;
}

int main()
{
	float* values;
	float* d_values;
	values = (float*)malloc(16 * 16 * sizeof(float));
	cudaMalloc(&d_values, 16 * 16 * sizeof(float));

	for (int i = 0; i < 16 * 16; i++)
	{
		values[i] = (float)i;
	}
	cudaMemcpy(d_values, values, 16 * 16 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 block(1);
	dim3 grid(1);

	setFather << <grid, block >> >(d_values, 16, 16);

	cudaMemcpy(values, d_values, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 16 * 16; i++)
	{
		std::cout << values[i] << std::endl;
	}

	free(values);
	cudaFree(d_values);

	return 0;
}