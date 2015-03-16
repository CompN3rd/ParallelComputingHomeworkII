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

//with respect to adaptive nature
//__global__ void computeHrizontalBathymetryFluxes_kernel(TreeArray* h, TreeArray* b, TreeArray* Bu, float g, int refinementBaseX, int refinementBaseY, int maxRecursions)
//{
//	int idxX = threadIdx.x + blockIdx.x * blockDim.x;
//	int idxY = threadIdx.y + blockIdx.y * blockDim.y;
//
//	uint2 BuExt = make_uint2(Bu->getWidth(), Bu->getHeight());
//	uint2 hExt = make_uint2(h->getWidth(), h->getHeight());
//
//	//compute rectangle of cells in Bu and Bv this kernel covers:
//	uint2 cellStart;
//	uint2 cellExt;
//	computeCellRectangle(BuExt, refinementBaseX, refinementBaseY, maxRecursions, idxX, idxY, cellStart, cellExt);
//
//	//check bounds
//	if (cellStart.x >= BuExt.x || cellStart.y >= BuExt.y)
//		return;
//
//	//check right end of the cell and it's neighbor, if we need to subdivide
//	int leftIndex = computeIndex(h->getWidth(), h->getHeight(), cellStart.x + cellExt.x - 1, cellStart.y);
//	int rightIndex = computeIndex(h->getWidth(), h->getHeight(), cellStart.x + cellExt.x, cellStart.y);
//	int leftDepth = h->getDepths()[leftIndex];
//	int rightDepth = h->getDepths()[rightIndex];
//	if (leftDepth == 0 && rightDepth == 0)
//	{
//		//we don't need to subdivide
//		//fill the whole rectangle with 0 depth
//		fillRect<int>(Bu->getDepths(), BuExt, 0, cellStart, cellExt);
//
//		//fill whole rectangle with 0 flow, except the right border
//		uint2 zeroStart = cellStart;
//		uint2 zeroExt = cellExt - make_uint2(1, 0);
//		fillRect<float>(Bu->getValues(), BuExt, 0.0f, zeroStart, zeroExt);
//
//		//fill the right border
//		float leftH = h->getValues()[leftIndex];
//		float rightH = h->getValues()[rightIndex];
//		float leftB = b->getValues()[leftIndex];
//		float rightB = b->getValues()[rightIndex];
//		float Bx = g*(rightH * rightB - leftH * leftB);
//
//		uint2 borderStart = cellStart;
//		borderStart.x += cellExt.x - 1;
//		uint2 borderExt = make_uint2(1, cellExt.y);
//		fillRect<float>(Bu->getValues(), BuExt, Bx, borderStart, borderExt);
//	}
//	else
//	{
//		//we need to subdivide
//	}
//}


//without respect to adaptive nature (works correct, too)
__global__ void computeBathymetrySources_kernel(TreeArray* h, TreeArray* b, TreeArray* Bu, TreeArray* Bv, float g, int maxRecursions)
{
	int idxX = threadIdx.x + blockIdx.x * blockDim.x;
	int idxY = threadIdx.y + blockIdx.y * blockDim.y;

	if (idxX >= Bu->getWidth() || idxY >= Bu->getHeight())
		return;

	int currentIndex = computeIndex(Bu->getWidth(), Bu->getHeight(), idxX, idxY);
	//this is the "left" or "bottom" index, note the different extends for these arrays
	int currentIndexH = computeIndex(h->getWidth(), h->getHeight(), idxX, idxY);
	//for vertical flux
	int topIndex = computeIndex(h->getWidth(), h->getHeight(), idxX, idxY + 1);
	//for horizontal flux
	int rightIndex = computeIndex(h->getWidth(), h->getHeight(), idxX + 1, idxY);

	Bu->getDepths()[currentIndex] = maxRecursions;
	Bv->getDepths()[currentIndex] = maxRecursions;

	Bu->getValues()[currentIndex] = g * (h->getValues()[rightIndex] * b->getValues()[rightIndex] - h->getValues()[currentIndexH] * b->getValues()[currentIndexH]);
	Bv->getValues()[currentIndex] = g * (h->getValues()[topIndex] * b->getValues()[topIndex] - h->getValues()[currentIndexH] * b->getValues()[currentIndexH]);
}

//--------------------------------
//fluxes

__global__ void computeFluxesF_kernel(TreeArray* h, TreeArray* hu, TreeArray* hv, TreeArray* Fh, TreeArray* Fhu, TreeArray* Fhv, float g)
{
	int idxX = threadIdx.x + blockIdx.x * blockDim.x;
	int idxY = threadIdx.y + blockIdx.y * blockDim.y;

	if (idxX >= Fh->getWidth() || idxY >= Fh->getHeight())
		return;

	int currentIndex = computeIndex(Fh->getWidth(), Fh->getHeight(), idxX, idxY);
	int currentIndexH = computeIndex(h->getWidth(), h->getHeight(), idxX, idxY);
	int rightIndexH = computeIndex(h->getWidth(), h->getHeight(), idxX + 1, idxY);

	//compute signal velocities
	float sv1, sv2, llf;
	sv1 = fabs(hu->getValues()[currentIndexH] / h->getValues()[currentIndexH]) + sqrtf(g * h->getValues()[currentIndexH]);
	sv2 = fabs(hu->getValues()[rightIndexH] / h->getValues()[rightIndexH]) + sqrtf(g * h->getValues()[rightIndexH]);
	llf = max(sv1, sv2);

	//compute fluxes:
	Fh->getValues()[currentIndex] = computeFlux(
		h->getValues()[currentIndexH] * hu->getValues()[currentIndexH],
		h->getValues()[rightIndexH] * hu->getValues()[rightIndexH],
		h->getValues()[currentIndexH],
		h->getValues()[rightIndexH],
		llf);

	Fhu->getValues()[currentIndex] = computeFlux(
		hu->getValues()[currentIndexH] * hu->getValues()[currentIndexH] + (0.5f * g * h->getValues()[currentIndexH]),
		hu->getValues()[rightIndexH] * hu->getValues()[rightIndexH] + (0.5f * g * h->getValues()[rightIndexH]),
		hu->getValues()[currentIndexH],
		hu->getValues()[rightIndexH],
		llf);

	Fhv->getValues()[currentIndex] = computeFlux(
		hu->getValues()[currentIndexH] * hv->getValues()[currentIndexH],
		hu->getValues()[rightIndexH] * hv->getValues()[rightIndexH],
		hv->getValues()[currentIndexH],
		hv->getValues()[rightIndexH],
		llf);
}

__global__ void computeFluxesG_kernel(TreeArray* h, TreeArray* hu, TreeArray* hv, TreeArray* Gh, TreeArray* Ghu, TreeArray* Ghv, float g)
{
	int idxX = threadIdx.x + blockIdx.x * blockDim.x;
	int idxY = threadIdx.y + blockIdx.y * blockDim.y;

	if (idxX >= Gh->getWidth() || idxY >= Gh->getHeight())
		return;

	int currentIndex = computeIndex(Gh->getWidth(), Gh->getHeight(), idxX, idxY);
	int currentIndexH = computeIndex(h->getWidth(), h->getHeight(), idxX, idxY);
	int topIndexH = computeIndex(h->getWidth(), h->getHeight(), idxX, idxY + 1);

	//compute signal velocities
	float sv1, sv2, llf;
	sv1 = fabs(hv->getValues()[currentIndexH] / h->getValues()[currentIndexH]) + sqrtf(g * h->getValues()[currentIndexH]);
	sv2 = fabs(hv->getValues()[topIndexH] / h->getValues()[topIndexH]) + sqrtf(g * h->getValues()[topIndexH]);
	llf = max(sv1, sv2);

	//compute fluxes
	Gh->getValues()[currentIndex] = computeFlux(
		h->getValues()[currentIndexH] * hv->getValues()[currentIndexH],
		h->getValues()[topIndexH] * hv->getValues()[topIndexH],
		h->getValues()[currentIndexH],
		h->getValues()[topIndexH],
		llf);

	Ghu->getValues()[currentIndex] = computeFlux(
		hu->getValues()[currentIndexH] * hv->getValues()[currentIndexH],
		hu->getValues()[topIndexH] * hv->getValues()[topIndexH],
		hu->getValues()[currentIndexH],
		hu->getValues()[topIndexH],
		llf);

	Ghv->getValues()[currentIndex] = computeFlux(
		hv->getValues()[currentIndexH] * hv->getValues()[currentIndexH] + (0.5f * g * h->getValues()[currentIndexH]),
		hv->getValues()[topIndexH] * hv->getValues()[topIndexH] + (0.5f * g * h->getValues()[topIndexH]),
		hv->getValues()[currentIndexH],
		hv->getValues()[topIndexH],
		llf);
}

//--------------------------------
//averaging of values
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