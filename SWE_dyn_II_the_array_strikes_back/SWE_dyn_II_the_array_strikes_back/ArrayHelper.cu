#include "ArrayHelper.h"

ArrayHelper::ArrayHelper(int x, int y)
{
	nx = x;
	ny = y;

	cudaMallocManaged(&values, nx * ny * sizeof(float));
	cudaMallocManaged(&depths, nx * ny * sizeof(int));
}

TreeArray::~TreeArray()
{
	//crashes???
	//cudaFree(values);
	//cudaFree(depths);
}