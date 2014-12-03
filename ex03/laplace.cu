#include <stdio.h>
#include <stdlib.h>

#include "writeVTK.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef _WIN32
#include "cuda_utils.h"
#else
#include <Windows.h>
struct Timer
{
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;

	double time = 0.0;
};

void startTimer(Timer& timer)
{
	if (QueryPerformanceFrequency(&timer.frequency) == FALSE)
	{
		fprintf(stderr, "QueryPerformanceFrequency failed \n");
		exit(1);
	}

	if (QueryPerformanceCounter(&timer.start) == FALSE)
	{
		fprintf(stderr, "QueryPerformanceCounter failed \n");
		exit(1);
	}
}

double stopTimer(Timer& timer)
{
	if (QueryPerformanceCounter(&timer.end) == FALSE)
	{
		fprintf(stderr, "QueryPerformanceCounter failed \n");
		exit(1);
	}

	return static_cast<double>(timer.end.QuadPart - timer.start.QuadPart) / timer.frequency.QuadPart;
}

#endif


#define NON_BOUND_ELEMENTS 512
#define BLOCK_SIZE 16
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

__global__ void laplaceTextureSharedStep(float* nextTime, int gridWidth, float alpha, float deltaT, float h)
{
	__shared__ float block[BLOCK_SIZE][BLOCK_SIZE];

	// shift indices by 1 -> larger Input Grid for boundary conditions
	int localRow = threadIdx.y;
	int localCol = threadIdx.x;
	int rowIndex = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int colIndex = blockIdx.x * blockDim.x + threadIdx.x + 1;

	//fetch data cooperatively in block
	block[localRow][localCol] = tex2D(inputTex, colIndex, rowIndex);
	__syncthreads();

	//fetch data from shared block except those over the edge
	float currentPos = block[localRow][localCol];

	float rightPos = 0;
	if (localCol + 1 < BLOCK_SIZE)
		rightPos = block[localRow][localCol + 1];
	else
		rightPos = tex2D(inputTex, colIndex + 1, rowIndex);

	float leftPos = 0;
	if (localCol - 1 >= 0)
		leftPos = block[localRow][localCol - 1];
	else
		leftPos = tex2D(inputTex, colIndex - 1, rowIndex);

	float topPos = 0;
	if (localRow - 1 >= 0)
		topPos = block[localRow - 1][localCol];
	else
		topPos = tex2D(inputTex, colIndex, rowIndex - 1);

	float bottomPos = 0;
	if (localRow + 1 < BLOCK_SIZE)
		bottomPos = block[localRow + 1][localCol];
	else
		bottomPos = tex2D(inputTex, colIndex, rowIndex + 1);

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
	Timer t;
	double time; 

	int arraySize = NON_BOUND_ELEMENTS + 2;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(NON_BOUND_ELEMENTS / block.x, NON_BOUND_ELEMENTS / block.y);

	float* h_currentTime = (float*)malloc(arraySize * arraySize * sizeof(float));
	float* h_nextTime = (float*)malloc(arraySize * arraySize * sizeof(float));

	//intialize arrays:
	initializeArrays(h_currentTime, h_nextTime, arraySize);

	//initialize device memory
	float* d_currentTime = NULL;
	float* d_nextTime = NULL;

	cudaMalloc((void**)&d_currentTime, arraySize * arraySize * sizeof(float));
	cudaMalloc((void**)&d_nextTime, arraySize * arraySize * sizeof(float));

	//copy
	cudaMemcpy(d_currentTime, h_currentTime, arraySize * arraySize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextTime, h_nextTime, arraySize * arraySize * sizeof(float), cudaMemcpyHostToDevice);

	time = 0.0;
	//loop over discrete time steps
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		//execute kernel with swapping input and output
		if (i % 2 == 0)
		{
			startTimer(t);
			laplaceGlobalStep << <grid, block >> >(d_currentTime, d_nextTime, arraySize, ALPHA, DELTA_T, H);

			//copy back to host
			cudaMemcpy(h_currentTime, d_currentTime, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToHost);
			time += stopTimer(t);

			//write to memory
			writeVTK("globalMem", i, arraySize, arraySize, h_currentTime);
		}
		else
		{
			startTimer(t);
			laplaceGlobalStep << <grid, block >> >(d_nextTime, d_currentTime, arraySize, ALPHA, DELTA_T, H);

			//copy back to host
			cudaMemcpy(h_nextTime, d_nextTime, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToHost);
			time += stopTimer(t);

			//write to memory
			writeVTK("globalMem", i, arraySize, arraySize, h_nextTime);
		}
	}

	printf("Global Memory Time: %f\n", time);

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

	time = 0.0;
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		startTimer(t);
		laplaceTextureStep << < grid, block >> >(d_nextTime, arraySize, ALPHA, DELTA_T, H);

		//copy result back to host:
		cudaMemcpyFromArray(h_currentTime, cuArray, 0, 0, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToHost);

		//copy result to array for next iteration
		cudaMemcpyToArray(cuArray, 0, 0, d_nextTime, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToDevice);
		time += stopTimer(t);

		//write output
		writeVTK("textureMem", i, arraySize, arraySize, h_currentTime);
	}

	printf("Texture Memory Time: %f\n", time);

	cudaUnbindTexture(inputTex);

	//texture + shared memory case
	initializeArrays(h_currentTime, h_nextTime, arraySize);

	//initializations as before
	cudaMallocArray(&cuArray, &channelDesc, arraySize, arraySize);
	cudaMemcpyToArray(cuArray, 0, 0, h_currentTime, arraySize * arraySize * sizeof(float), cudaMemcpyHostToDevice);

	cudaBindTextureToArray(inputTex, cuArray, channelDesc);

	time = 0;
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		startTimer(t);
		laplaceTextureSharedStep << < grid, block >> >(d_nextTime, arraySize, ALPHA, DELTA_T, H);

		//copy result back to host:
		cudaMemcpy(h_currentTime, d_nextTime, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToHost);

		//copy result to array for next iteration
		cudaMemcpyToArray(cuArray, 0, 0, d_nextTime, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToDevice);
		time += stopTimer(t);

		//write output
		writeVTK("textureSharedMem", i, arraySize, arraySize, h_currentTime);
	}

	printf("Texture Shared Memory Time: %f\n", time);

	cudaUnbindTexture(inputTex);

	//texture + no copy version
	free(h_nextTime);
	cudaFree(d_nextTime);

	// make h_nextTime and d_nextTime big enough to hold all results
	h_nextTime = (float*)malloc(NUM_ITERATIONS * arraySize * arraySize * sizeof(float));
	cudaMalloc((void**)&d_nextTime, NUM_ITERATIONS * arraySize * arraySize * sizeof(float));

	initializeArrays(h_currentTime, h_nextTime, arraySize);
	cudaMemcpyToArray(cuArray, 0, 0, h_currentTime, arraySize * arraySize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(d_nextTime, 0, NUM_ITERATIONS * arraySize * arraySize * sizeof(float));

	cudaBindTextureToArray(inputTex, cuArray, channelDesc);

	time = 0;
	startTimer(t);
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		laplaceTextureStep << <grid, block >> >(d_nextTime + i * arraySize * arraySize, arraySize, ALPHA, DELTA_T, H);

		//don't copy back to host, just to cuArray for next iteration
		cudaMemcpyToArray(cuArray, 0, 0, d_nextTime + i * arraySize * arraySize, arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	//download all results:
	cudaMemcpy(h_nextTime, d_nextTime, NUM_ITERATIONS * arraySize * arraySize * sizeof(float), cudaMemcpyDeviceToHost);

	time += stopTimer(t);
	printf("Texture + No Copy Memory Time: %f\n", time);

	//output all results
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		writeVTK("textureNoCopyMem", i, arraySize, arraySize, h_nextTime + i * arraySize * arraySize);
	}

	// free memory
	cudaUnbindTexture(inputTex);
	cudaFreeArray(cuArray);
	free(h_currentTime);
	free(h_nextTime);
	cudaFree(d_nextTime);
	cudaFree(d_currentTime);

	cudaDeviceReset();

	return 0;
}