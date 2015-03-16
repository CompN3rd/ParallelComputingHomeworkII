#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <math.h>
#include <helper_math.h>

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
		return this->nx;
	}

	inline __device__ __host__ int getHeight()
	{
		return this->ny;
	}

	inline __device__ __host__ float* const getValues()
	{
		return this->values;
	}

	inline __device__ __host__ int* const getDepths()
	{
		return this->depths;
	}
};