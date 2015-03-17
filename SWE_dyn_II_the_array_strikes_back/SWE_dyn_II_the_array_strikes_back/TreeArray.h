#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define __DRIVER_TYPES_H__
#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_math.h>
#include <iostream>
#include <stdio.h>
#include <fstream>

class Managed
{
public:

	void *operator new(size_t len)
	{
		void* ptr;
		cudaMallocManaged(&ptr, len);
		return ptr;
	}

	void operator delete(void* ptr)
	{
		cudaFree(ptr);
	}
};

class TreeArray : public Managed
{
private:
	int nx;
	int ny;

	//array for values
	float* values;

	//array, indicating the level of the tree
	int* depths;

public:
	TreeArray(int x, int y);
	//unified memory copy constructor
	TreeArray(const TreeArray& arr);
	~TreeArray();

	//element access
	inline __device__ __host__ int getWidth()
	{
		cudaDeviceSynchronize();
		return this->nx;
	}

	inline __device__ __host__ int getHeight()
	{
		cudaDeviceSynchronize();
		return this->ny;
	}

	inline __device__ __host__ float* getValues()
	{
		cudaDeviceSynchronize();
		return this->values;
	}

	inline __device__ __host__ int* getDepths()
	{
		cudaDeviceSynchronize();
		return this->depths;
	}
};