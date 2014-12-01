#ifndef _WIN32
#include "cuda_utils.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "writeVTK.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NON_BOUND_ELEMENTS 512
#define GRID_SIZE 16
#define NUM_ITERATIONS 20
#define ALPHA 0.1f
#define DELTA_T 1.0f
#define H 1.0f

// texture object declarations
texture<float, cudaTextureType2D, cudaReadModeElementType> inputTex;

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

__global__ void laplaceTextureStep(float* nextTime, int gridWidth, float alpha, float deltaT, float h)
{
	// shift indices by 1 -> larger Input Grid for boundary conditions
	int rowIndex = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int colIndex = blockIdx.x * blockDim.x + threadIdx.x + 1;

	//fetch data from inputTex
	float currentPos = tex2D(inputTex, colIndex, rowIndex);
	float rightPos = tex2D(inputTex, colIndex + 1, rowIndex);
	float leftPos = tex2D(inputTex, colIndex - 1, rowIndex);
	float topPos = tex2D(inputTex, colIndex, rowIndex - 1);
	float bottomPos = tex2D(inputTex, colIndex, rowIndex + 1);

	//writing back same as in global memory kernel
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
	dim3 block(16,16);
	dim3 grid(NON_BOUND_ELEMENTS / block.x, NON_BOUND_ELEMENTS / block.y);

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

	//loop over discrete time steps
	for(int i = 0; i < NUM_ITERATIONS; i++)
	{
		//execute kernel with swapping input and output
		if(i % 2 == 0)
		{
			laplaceGlobalStep << <grid, block >> >(d_currentTime, d_nextTime, arraySize, ALPHA, DELTA_T, H);

			//copy back to host
			cudaMemcpy(h_currentTime, d_currentTime, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToHost);

			//write to memory
			writeVTK("globalMem", i, arraySize, arraySize, h_currentTime);
		}
		else
		{
			laplaceGlobalStep << <grid, block >> >(d_nextTime, d_currentTime, arraySize, ALPHA, DELTA_T, H);

			//copy back to host
			cudaMemcpy(h_nextTime, d_nextTime, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToHost);

			//write to memory
			writeVTK("globalMem", i, arraySize, arraySize, h_nextTime);
		}
	}

	//texture memory case
	initializeArrays(h_currentTime, h_nextTime, arraySize);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray* cuArray;
	cudaMallocArray(&cuArray, &channelDesc, arraySize, arraySize);

	cudaMemcpyToArray(cuArray, 0, 0, h_currentTime, arraySize * arraySize * sizeof(float), cudaMemcpyHostToDevice);

	inputTex.addressMode[0] = cudaAddressModeClamp;
	inputTex.addressMode[1] = cudaAddressModeClamp;
	inputTex.filterMode = cudaFilterModePoint;
	inputTex.normalized = false;

	cudaBindTextureToArray(inputTex, cuArray, channelDesc);

	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		laplaceTextureStep << < grid, block >> >(d_nextTime, arraySize, ALPHA, DELTA_T, H);

		//copy result back to host:
		cudaMemcpyFromArray(h_currentTime, cuArray, 0, 0, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToHost);

		//copy result to array for next iteration
		cudaMemcpyToArray(cuArray, 0, 0, d_nextTime, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToDevice);

		//write output
		writeVTK("textureMem", i, arraySize, arraySize, h_currentTime);
	}


	// free memory
	free(h_currentTime);
	free(h_nextTime);
	cudaFree(d_nextTime);
	cudaFree(d_currentTime);

	cudaDeviceReset();

	return 0;
}