#include "TreeArray.h"

typedef enum BoundaryType {
	OUTFLOW, WALL, CONNECT
} BoundaryType;

//row wise memory layout
inline __device__ __host__ int computeIndex(int width, int height, int x0, int y0)
{
	// no bounds check!!!
	return x0 + y0 * width;
}

inline __device__ __host__ int computeIndex(uint2 ext, uint2 p)
{
	// no bounds check!!!
	return p.x + p.y * ext.x;
}

//blocksize computation
inline __device__ __host__ unsigned int divUp(const unsigned int i, const unsigned int d)
{
	return ((i + d - 1) / d);
}

template<typename MemoryType>
__global__ void fillRect_child(MemoryType* data, const uint2& arrayExtends, MemoryType value, const uint2& pStart, const uint2& rectExt)
{
	//assumption gridSize = rectExt.(width, height)
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y;

	if (xIndex >= rectExt.x || yIndex >= rectExt.y)
		return;

	int x0 = pStart.x + xIndex;
	int y0 = pStart.y + yIndex;

	if (x0 < 0 || x0 >= arrayExtends.x || y0 < 0 || y0 >= arrayExtends.y)
		return;

	data[computeIndex(arrayExtends.x, arrayExtends.y, x0, y0)] = value;
}

template<typename MemoryType>
__device__ void fillRect(MemoryType* data, const uint2& arrayExtends, MemoryType value, const uint2& pStart, const uint2& rectExt)
{
	dim3 block(min(rectExt.x, 16), min(rectExt.y, 16));
	dim3 grid(divUp(rectExt.x, block.x), divUp(rectExt.y, block.y));
	fillRect_child<MemoryType> << <grid, block >> >(data, arrayExtends, value, pStart, rectExt);
}

//thread computation
inline __device__ __host__ unsigned int computeForestBase(unsigned int dimension, unsigned int base, unsigned int maxRecursions)
{
	return divUp(dimension, (unsigned int)pow(base, maxRecursions));
}

//cell rectangle
inline __device__ __host__ void computeCellRectangle(const uint2 ext, unsigned int refinementBaseX, unsigned int refinementBaseY, unsigned int refinementHeight, unsigned int idxX, unsigned int idxY, uint2& cellStart, uint2& cellExt)
{
	unsigned int baseLengthX = (unsigned int)pow(refinementBaseX, refinementHeight);
	unsigned int baseLengthY = (unsigned int)pow(refinementBaseY, refinementHeight);

	unsigned int x = min(ext.x, idxX * baseLengthX);
	unsigned int y = min(ext.y, idxY * baseLengthY);
	cellStart.x = x;
	cellStart.y = y;

	cellExt.x = x + baseLengthX < ext.x ? baseLengthX : 0;
	cellExt.y = y + baseLengthY < ext.y ? baseLengthY : 0;
}

// local Lax-Friedrich
inline __device__ float computeFlux(float fLow, float fHigh, float xiLow, float xiHigh, float llf) 
{
	return 0.5f*(fLow + fHigh) - 0.5f*llf*(xiHigh - xiLow);
}
