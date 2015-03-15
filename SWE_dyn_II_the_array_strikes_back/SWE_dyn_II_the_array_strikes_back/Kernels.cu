#include "Kernels.h"

__global__ void setTopBorder_kernel(TreeArray* h, TreeArray* hu, TreeArray* hv, BoundaryType top)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= h->getWidth())
		return;

	if (top == CONNECT)
	{
		h->getValues()[computeIndex(h->getWidth(), h->getHeight(), idx, h->getHeight() - 1)] = h->getValues()[computeIndex(h->getWidth(), h->getHeight(), idx, 1)];
		hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), idx, hu->getHeight() - 1)] = hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), idx, 1)];
		hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, hv->getHeight() - 1)] = hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, 1)];
	}
	else
	{
		h->getValues()[computeIndex(h->getWidth(), h->getHeight(), idx, h->getHeight() - 1)] = h->getValues()[computeIndex(h->getWidth(), h->getHeight(), idx, h->getHeight() - 2)];
		hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), idx, hu->getHeight() - 1)] = hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), idx, hu->getHeight() - 2)];
		hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, hv->getHeight() - 1)] = (top == WALL) ? -hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, hv->getHeight() - 2)] : hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, hv->getHeight() - 2)];
	}
}

__global__ void setBottomBorder_kernel(TreeArray* h, TreeArray* hu, TreeArray* hv, BoundaryType bottom)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= h->getWidth())
		return;

	if (bottom == CONNECT)
	{
		h->getValues()[computeIndex(h->getWidth(), h->getHeight(), idx, 0)] = h->getValues()[computeIndex(h->getWidth(), h->getHeight(), idx, h->getHeight() - 2)];
		hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), idx, 0)] = hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), idx, hu->getHeight() - 2)];
		hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, 0)] = hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, hv->getHeight() - 2)];
	}
	else
	{
		h->getValues()[computeIndex(h->getWidth(), h->getHeight(), idx, 0)] = h->getValues()[computeIndex(h->getWidth(), h->getHeight(), idx, 1)];
		hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), idx, 0)] = hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), idx, 1)];
		hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, 0)] = (bottom == WALL) ? -hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, 1)] : hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), idx, 1)];
	}
}

__global__ void setRightBorder_kernel(TreeArray* h, TreeArray* hu, TreeArray* hv, BoundaryType right)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= h->getHeight())
		return;

	if (right = CONNECT)
	{
		h->getValues()[computeIndex(h->getWidth(), h->getHeight(), h->getWidth() - 1, idx)] = h->getValues()[computeIndex(h->getWidth(), h->getHeight(), 1, idx)];
		hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), hu->getWidth() - 1, idx)] = hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), 1, idx)];
		hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), hv->getWidth() - 1, idx)] = hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), 1, idx)];
	}
	else
	{
		h->getValues()[computeIndex(h->getWidth(), h->getHeight(), h->getWidth() - 1, idx)] = h->getValues()[computeIndex(h->getWidth(), h->getHeight(), h->getWidth() - 2, idx)];
		hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), hu->getWidth() - 1, idx)] = (right == WALL) ? -hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), hu->getWidth() - 2, idx)] : hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), hu->getWidth() - 2, idx)];
		hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), hv->getWidth() - 1, idx)] = hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), hv->getWidth() - 2, idx)];
	}
}

__global__ void setLeftBorder_kernel(TreeArray* h, TreeArray* hu, TreeArray* hv, BoundaryType left)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= h->getHeight())
		return;

	if (left = CONNECT)
	{
		h->getValues()[computeIndex(h->getWidth(), h->getHeight(), 0, idx)] = h->getValues()[computeIndex(h->getWidth(), h->getHeight(), h->getWidth() - 2, idx)];
		hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), 0, idx)] = hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), h->getWidth() - 2, idx)];
		hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), 0, idx)] = hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), h->getWidth() - 2, idx)];
	}
	else
	{
		h->getValues()[computeIndex(h->getWidth(), h->getHeight(), 0, idx)] = h->getValues()[computeIndex(h->getWidth(), h->getHeight(), 1, idx)];
		hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), 0, idx)] = (left == WALL) ? -hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), 1, idx)] : hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), 1, idx)];
		hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), 0, idx)] = hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), 1, idx)];
	}
}

//--------------------------------
//bathymetry
__global__ void computeBathymetry_kernel(TreeArray* h, TreeArray* b, TreeArray* Bu, TreeArray* Bv, float g, int refinementBaseX, int refinementBaseY, int maxRecursions)
{
	int idxX = threadIdx.x + blockIdx.x * blockDim.x;
	int idxY = threadIdx.y + blockIdx.y * blockDim.y;

	uint2 BuExt = make_uint2(Bu->getWidth(), Bu->getHeight());
	uint2 hExt = make_uint2(h->getWidth(), h->getHeight());

	uint2 cellStart;
	uint2 cellExt;
	computeCellRectangle(BuExt, refinementBaseX, refinementBaseY, maxRecursions, idxX, idxY, cellStart, cellExt);

	if (cellStart.x >= BuExt.x || cellStart.y >= BuExt.y)
		return;

	//check starting pos, if we need to subdivide
	int targetDepth = h->getDepths()[computeIndex(h->getWidth(), h->getHeight(), cellStart.x + 1, cellStart.y + 1)];
	if (targetDepth == 0)
	{
		//we don't need to subdivide
	}
	else
	{
		//we need to subdivide
	}
}

//--------------------------------
__global__ void getAveragedVerticalValue_child_kernel(TreeArray* arr, uint2 globStart, int refinementBase, int myDepth, float* resFather, int* numFather)
{
	int idxY = threadIdx.x + blockIdx.x * blockDim.x;
	int baseLengthY = (int)pow(refinementBase, maxRecursions - myDepth);
}

//averaging part of a row / column
__device__ float getAveragedVerticalValue(TreeArray* arr, uint2 start, int refinementBase, int myDepth)
{
	//check depth of start, if refinement is needed
	if (arr->getDepths()[computeIndex(arr->getWidth(), arr->getHeight(), start.x, start.y)] == myDepth)
	{
		//no subdivision
		//value at start is already the averaged value
		return arr->getValues()[computeIndex(arr->getWidth(), arr->getHeight(), start.x, start.y)];
	}
	else
	{
		//array for sub-averages
		float* subSums = new float[refinementBase];
		int* numElems = new int[refinementBase];

		//fill those arrays recursively
		dim3 block(refinementBase);
		dim3 grid(1);
		getAveragedVerticalValue_child_kernel << <grid, block >> >(arr, start, refinementBase, myDepth + 1, subSums, numElems);

		//sum them up
		float res = 0;
		int num = 0;
		for (int i = 0; i < refinementBase; i++)
		{
			res += subSums[i];
			num += numElems[i];
		}

		delete[] subSums;
		delete[] numElems;

		return res / num;
	}
}

__device__ float getAveragedHorizontalValue(TreeArray* arr, uint2 start, int refinementBase, int myDepth)
{

}