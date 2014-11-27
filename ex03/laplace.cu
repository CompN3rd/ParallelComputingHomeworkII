#include "cuda_utils.h"
#include "writeVTK.h"
#include <stdio.h>
#include <stdlib.h>

#define NON_BOUND_ELEMENTS 2048
#define GRID_SIZE 16

__device__ __host__ inline int getLinearIndex(int rowIndex, int colIndex, int arrayWidth)
{
	return rowIndex * gridWidth + colIndex;
}

__global__ void laplaceGlobalStep(float* currentTime, float* nextTime, int gridWidth, float alpha, float deltaT, float h)
{
	// shift indices by 1 -> larger Input Grid for boundary conditions
	int rowIndex = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int colIndex = blockIdx.x * blockDim.x + threadIdx.x + 1;

	// gridWidth includes boundary elements!
	float currentPos = currrentTime[getLinearIndex(rowIndex, colIndex, gridWidth)];
	float rightPos = currrentTime[getLinearIndex(rowIndex, colIndex + 1, gridWidth)];
	float leftPos = currrentTime[getLinearIndex(rowIndex, colIndex - 1, gridWidth)];
	float topPos = currrentTime[getLinearIndex(rowIndex - 1, colIndex, gridWidth)];
	float bottomPos = currrentTime[getLinearIndex(rowIndex + 1, colIndex, gridWidth)];

	// explicit finite differences formula
	nextTime[getLinearIndex(rowIndex, colIndex, gridWidth)] = currentPos + alpha * deltaT / (h * h) * (topPos + bottomPos + rightPos + leftPos - 4 * currentPos);
} 

int main()
{
	//global memory part
	int arraySize = NON_BOUND_ELEMENTS + 2;

	float* h_currentTime = (float*) malloc(arraySize * arraySize * sizeof(float));
	float* h_nextTime = (float*) malloc(arraySize * arraySize * sizeof(float));
	float* d_currentTime = NULL;
	float* d_nextTime = NULL;

	cudaMalloc((void**) &d_currentTime, arraySize * arraySize * sizeof(float));

		

	return 0;
}