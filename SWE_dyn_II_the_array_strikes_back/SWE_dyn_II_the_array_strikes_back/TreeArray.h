#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <math.h>

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
public:
	int nx;
	int ny;

	//array for values
	float* values;

	//array, indicating the level of the tree
	int* depths;

	TreeArray(int x, int y);
	//unified memory copy constructor
	TreeArray(const TreeArray& arr);
	~TreeArray();
};