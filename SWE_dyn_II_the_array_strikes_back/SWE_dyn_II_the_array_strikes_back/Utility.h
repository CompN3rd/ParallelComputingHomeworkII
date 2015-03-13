#include "TreeArray.h"

//row wise memory layout
inline __device__ __host__ int computeIndex(int width, int height, int x0, int y0)
{
	// no bounds check!!!
	return x0 + y0 * width;
}

//blocksize computation
inline __device__ __host__ unsigned int divUp(const unsigned int i, const unsigned int d)
{
	return ((i + d - 1) / d);
}

template<typename MemoryType>
__global__ void fillRect(MemoryType* data, int arrayWidth, int arrayHeight, MemoryType value, int x0, int y0, int width, int height)
{
	//assumption gridSize = (width, height)
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex >= width || yIndex >= height)
		return;

	x0 += xIndex;
	y0 += yIndex;

	if (x0 < 0 || x0 >= arrayWidth || y0 < 0 || y0 >= arrayHeight)
		return;

	data[computeIndex(arrayWidth, arrayHeight, x0, y0)] = value;
}