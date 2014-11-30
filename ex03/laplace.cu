#ifndef _WIN32
#include "cuda_utils.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "writeVTK.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NON_BOUND_ELEMENTS 2048
#define GRID_SIZE 16
#define NUM_ITERATIONS 20

__device__ __host__ inline int getLinearIndex(int rowIndex, int colIndex, int arrayWidth)
{
	return rowIndex * arrayWidth + colIndex;
}

__global__ void laplaceGlobalStep(float* currentTime, float* nextTime, int gridWidth, float alpha, float deltaT, float h)
{
	// shift indices by 1 -> larger Input Grid for boundary conditions
	int rowIndex = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int colIndex = blockIdx.x * blockDim.x + threadIdx.x + 1;

	// gridWidth includes boundary elements!
	float currentPos = currentTime[getLinearIndex(rowIndex, colIndex, gridWidth)];
	float rightPos = currentTime[getLinearIndex(rowIndex, colIndex + 1, gridWidth)];
	float leftPos = currentTime[getLinearIndex(rowIndex, colIndex - 1, gridWidth)];
	float topPos = currentTime[getLinearIndex(rowIndex - 1, colIndex, gridWidth)];
	float bottomPos = currentTime[getLinearIndex(rowIndex + 1, colIndex, gridWidth)];

	// explicit finite differences formula
	nextTime[getLinearIndex(rowIndex, colIndex, gridWidth)] = currentPos + alpha * deltaT / (h * h) * (topPos + bottomPos + rightPos + leftPos - 4 * currentPos);
} 

void initializeArrays(float* currentTime, float* nextTime, int arrayWidth)
{
	//initialize host memory:
	for (int i = 0; i < arrayWidth; i++)
	{
		for (int j = 0; j < arrayWidth; j++)
		{
			if (i >= 3 * arrayWidth / 8 && i < 5 * arrayWidth / 8 && j >= arrayWidth / 3 && j < 2 * arrayWidth / 3)
			{
				currentTime[i * arrayWidth + j] = 100;
				nextTime[i * arrayWidth + j] = 100;
			}
			else
			{
				currentTime[i * arrayWidth + j] = 0;
				nextTime[i * arrayWidth + j] = 0;
			}
		}
	}
}

int main()
{
	int arraySize = NON_BOUND_ELEMENTS + 2;

	float* h_currentTime = (float*) malloc(arraySize * arraySize * sizeof(float));
	float* h_nextTime = (float*) malloc(arraySize * arraySize * sizeof(float));

	//intialize arrays:
	initializeArrays(h_currentTime, h_nextTime, arraySize);

	//initialize device memory
	float* d_currentTime = NULL;
	float* d_nextTime = NULL;

	cudaMalloc((void**) &d_currentTime, arraySize * arraySize * sizeof(float));
	cudaMalloc((void**) &d_nextTime, arraySize * arraySize * sizeof(float));

	//copy
	cudaMemcpy(d_currentTime, h_currentTime, arraySize * arraySize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextTime, h_nextTime, arraySize * arraySize * sizeof(float), cudaMemcpyHostToDevice);

	return 0;
}