#include "TreeArray.h"

TreeArray::TreeArray(int x, int y)
{
	nx = x;
	ny = y;

	cudaMallocManaged(&values, nx * ny * sizeof(float));
	cudaMallocManaged(&depths, nx * ny * sizeof(int));
}

TreeArray::TreeArray(const TreeArray& other)
{
	nx = other.nx;
	ny = other.ny;

	cudaMallocManaged(&values, nx * ny * sizeof(float));
	cudaMallocManaged(&depths, nx * ny * sizeof(float));

	memcpy(values, other.values, nx * ny * sizeof(float));
	memcpy(depths, other.depths, nx * ny * sizeof(float));
}

TreeArray::~TreeArray()
{
	//crashes???
	//cudaFree(values);
	//cudaFree(depths);
}