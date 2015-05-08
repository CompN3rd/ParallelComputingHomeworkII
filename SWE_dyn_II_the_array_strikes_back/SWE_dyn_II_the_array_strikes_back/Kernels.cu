#include "Kernels.h"

__global__ void setTopBorder_kernel(float* h, float* hu, float* hv, int sizeX, int sizeY, BoundaryType top)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= sizeX + 2)
		return;

	if (top == CONNECT)
	{
		h[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY + 1)] = h[computeIndex(sizeX + 2, sizeY + 2, idx, 1)];
		hu[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY + 1)] = h[computeIndex(sizeX + 2, sizeY + 2, idx, 1)];
		hv[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY + 1)] = h[computeIndex(sizeX + 2, sizeY + 2, idx, 1)];
	}
	else
	{
		h[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY + 1)] = h[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY)];
		hu[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY + 1)] = hu[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY)];
		hv[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY + 1)] = (top == WALL) ? -hv[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY)] : hv[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY)];
	}
}

__global__ void setBottomBorder_kernel(float* h, float* hu, float* hv, int sizeX, int sizeY, BoundaryType bottom)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= sizeX + 2)
		return;

	if (bottom == CONNECT)
	{
		h[computeIndex(sizeX + 2, sizeY + 2, idx, 0)] = h[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY)];
		hu[computeIndex(sizeX + 2, sizeY + 2, idx, 0)] = hu[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY)];
		hv[computeIndex(sizeX + 2, sizeY + 2, idx, 0)] = hv[computeIndex(sizeX + 2, sizeY + 2, idx, sizeY)];
	}
	else
	{
		h[computeIndex(sizeX + 2, sizeY + 2, idx, 0)] = h[computeIndex(sizeX + 2, sizeY + 2, idx, 1)];
		hu[computeIndex(sizeX + 2, sizeY + 2, idx, 0)] = hu[computeIndex(sizeX + 2, sizeY + 2, idx, 1)];
		hv[computeIndex(sizeX + 2, sizeY + 2, idx, 0)] = (bottom == WALL) ? -hv[computeIndex(sizeX + 2, sizeY + 2, idx, 1)] : hv[computeIndex(sizeX + 2, sizeY + 2, idx, 1)];
	}
}

__global__ void setRightBorder_kernel(float* h, float* hu, float* hv, int sizeX, int sizeY, BoundaryType right)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= sizeY + 2)
		return;

	if (right == CONNECT)
	{
		h[computeIndex(sizeX + 2, sizeY + 2, sizeX +  1, idx)] = h[computeIndex(sizeX + 2, sizeY + 2, 1, idx)];
		hu[computeIndex(sizeX + 2, sizeY + 2, sizeX +  1, idx)] = hu[computeIndex(sizeX + 2, sizeY + 2, 1, idx)];
		hv[computeIndex(sizeX + 2, sizeY + 2, sizeX +  1, idx)] = hv[computeIndex(sizeX + 2, sizeY + 2, 1, idx)];
	}
	else
	{
		h[computeIndex(sizeX + 2, sizeY + 2, sizeX + 1, idx)] = h[computeIndex(sizeX + 2, sizeY + 2, sizeX, idx)];
		hu[computeIndex(sizeX + 2, sizeY + 2, sizeX + 1, idx)] = (right == WALL) ? -hu[computeIndex(sizeX + 2, sizeY + 2, sizeX, idx)] : hu[computeIndex(sizeX + 2, sizeY + 2, sizeX, idx)];
		hv[computeIndex(sizeX + 2, sizeY + 2, sizeX + 1, idx)] = hv[computeIndex(sizeX + 2, sizeY + 2, sizeX, idx)];
	}
}

__global__ void setLeftBorder_kernel(float* h, float* hu, float* hv, int sizeX, int sizeY, BoundaryType left)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= sizeY + 2)
		return;

	if (left == CONNECT)
	{
		h[computeIndex(sizeX + 2, sizeY + 2, 0, idx)] = h[computeIndex(sizeX + 2, sizeY + 2, sizeX, idx)];
		hu[computeIndex(sizeX + 2, sizeY + 2, 0, idx)] = hu[computeIndex(sizeX + 2, sizeY + 2, sizeX, idx)];
		hv[computeIndex(sizeX + 2, sizeY + 2, 0, idx)] = hv[computeIndex(sizeX + 2, sizeY + 2, sizeX, idx)];
	}
	else
	{
		h[computeIndex(sizeX + 2, sizeY + 2, 0, idx)] = h[computeIndex(sizeX + 2, sizeY + 2, 1, idx)];
		hu[computeIndex(sizeX + 2, sizeY + 2, 0, idx)] = (left == WALL) ? -hu[computeIndex(sizeX + 2, sizeY + 2, 1, idx)] : hu[computeIndex(sizeX + 2, sizeY + 2, 1, idx)];
		hv[computeIndex(sizeX + 2, sizeY + 2, 0, idx)] = hv[computeIndex(sizeX + 2, sizeY + 2, 1, idx)];
	}
}

//--------------------------------
//bathymetry

//with respect to adaptive nature
//__global__ void computeHrizontalBathymetryFluxes_kernel(float* h, float* b, float* Bu, float g, int refinementBaseX, int refinementBaseY, int maxRecursions)
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
__global__ void computeBathymetrySources_kernel(float* h, float* b, float* Bu, float* Bv, int sizeX, int sizeY, float g, int maxRecursions)
{
	int idxX = threadIdx.x + blockIdx.x * blockDim.x;
	int idxY = threadIdx.y + blockIdx.y * blockDim.y;

	if (idxX >= sizeX + 1 || idxY >= sizeY + 1)
		return;

	int currentIndex = computeIndex(sizeX + 2, sizeY + 2, idxX, idxY);
	int topIndex = computeIndex(sizeX + 2, sizeY + 2, idxX, idxY + 1);
	int rightIndex = computeIndex(sizeX + 2, sizeY + 2, idxX + 1, idxY);

	Bu[currentIndex] = g * (h[rightIndex] * b[rightIndex] - h[currentIndex] * b[currentIndex]);
	Bv[currentIndex] = g * (h[topIndex] * b[topIndex] - h[currentIndex] * b[currentIndex]);
}

//--------------------------------
//fluxes

__global__ void computeFluxesF_kernel(float* h, float* hu, float* hv, float* Fh, float* Fhu, float* Fhv, int sizeX, int sizeY, float g)
{
	int idxX = threadIdx.x + blockIdx.x * blockDim.x;
	int idxY = threadIdx.y + blockIdx.y * blockDim.y;

	if (idxX >= sizeX + 1 || idxY >= sizeY + 1)
		return;

	int currentIndex = computeIndex(sizeX + 1, sizeY + 1, idxX, idxY);
	int currentIndexH = computeIndex(sizeX + 2, sizeY + 2, idxX, idxY);
	int rightIndexH = computeIndex(sizeX + 2, sizeY + 2, idxX + 1, idxY);

	//compute signal velocities
	float sv1, sv2, llf;
	sv1 = fabs(hu[currentIndexH] / h[currentIndexH]) + sqrtf(g * h[currentIndexH]);
	sv2 = fabs(hu[rightIndexH] / h[rightIndexH]) + sqrtf(g * h[rightIndexH]);
	llf = max(sv1, sv2);

	//compute fluxes:
	Fh[currentIndex] = computeFlux(
		h[currentIndexH] * hu[currentIndexH],
		h[rightIndexH] * hu[rightIndexH],
		h[currentIndexH],
		h[rightIndexH],
		llf);

	Fhu[currentIndex] = computeFlux(
		hu[currentIndexH] * hu[currentIndexH] + (0.5f * g * h[currentIndexH]),
		hu[rightIndexH] * hu[rightIndexH] + (0.5f * g * h[rightIndexH]),
		hu[currentIndexH],
		hu[rightIndexH],
		llf);

	Fhv[currentIndex] = computeFlux(
		hu[currentIndexH] * hv[currentIndexH],
		hu[rightIndexH] * hv[rightIndexH],
		hv[currentIndexH],
		hv[rightIndexH],
		llf);
}

__global__ void computeFluxesG_kernel(float* h, float* hu, float* hv, float* Gh, float* Ghu, float* Ghv, int sizeX, int sizeY, float g)
{
	int idxX = threadIdx.x + blockIdx.x * blockDim.x;
	int idxY = threadIdx.y + blockIdx.y * blockDim.y;

	if (idxX >= sizeX + 1 || idxY >= sizeY + 1)
		return;

	int currentIndex = computeIndex(sizeX + 1, sizeY + 1, idxX, idxY);
	int currentIndexH = computeIndex(sizeX + 2, sizeY + 2, idxX, idxY);
	int topIndexH = computeIndex(sizeX + 2, sizeY + 2, idxX, idxY + 1);

	//compute signal velocities
	float sv1, sv2, llf;
	sv1 = fabs(hv[currentIndexH] / h[currentIndexH]) + sqrtf(g * h[currentIndexH]);
	sv2 = fabs(hv[topIndexH] / h[topIndexH]) + sqrtf(g * h[topIndexH]);
	llf = max(sv1, sv2);

	//compute fluxes
	Gh[currentIndex] = computeFlux(
		h[currentIndexH] * hv[currentIndexH],
		h[topIndexH] * hv[topIndexH],
		h[currentIndexH],
		h[topIndexH],
		llf);

	Ghu[currentIndex] = computeFlux(
		hu[currentIndexH] * hv[currentIndexH],
		hu[topIndexH] * hv[topIndexH],
		hu[currentIndexH],
		hu[topIndexH],
		llf);

	Ghv[currentIndex] = computeFlux(
		hv[currentIndexH] * hv[currentIndexH] + (0.5f * g * h[currentIndexH]),
		hv[topIndexH] * hv[topIndexH] + (0.5f * g * h[topIndexH]),
		hv[currentIndexH],
		hv[topIndexH],
		llf);
}

//--------------------------------
//euler timestep
__global__ void eulerTimestep_child_kernel(float* h, float* hu, float* hv,
	float* Fh, float* Fhu, float* Fhv,
	float* Gh, float* Ghu, float* Ghv,
	float* Bu, float* Bv,
	int* tree,
	int sizeX, int sizeY,
	float dt, float dxi, float dyi,
	int refinementBaseX, int refinementBaseY, int maxRecursions,
	int depth, uint2 cellStart, uint2 cellExt)
{
	int cellOffX = threadIdx.x + blockIdx.x * blockDim.x;
	int cellOffY = threadIdx.y + blockIdx.y * blockDim.y;
	uint2 offset = make_uint2(1, 1);

	uint2 hExt = make_uint2(sizeX + 2, sizeY + 2);
	uint2 ccellStart;
	uint2 ccellExt;
	computeCellRectangle(cellExt, refinementBaseX, refinementBaseY, maxRecursions - depth, cellOffX, cellOffY, ccellStart, ccellExt);

	if (ccellStart.x >= cellExt.x || ccellStart.y >= cellExt.y)
		return;

	//modify cellStart to the global compute grid grid
	cellStart += ccellStart;
	//modify cellExt to the real cell extends
	cellExt = ccellExt;

	if (depth == maxRecursions)
	{
		//we can't subdivide anymore, already at finest grid level; don't forget offset
		int currentIndexH = computeIndex(hExt, cellStart + offset);
		int currentIndex = computeIndex(sizeX + 1, sizeY + 1, cellStart.x + offset.x, cellStart.y + offset.y);
		int leftIndex = computeIndex(sizeX + 1, sizeY + 1, cellStart.x + offset.x - 1, cellStart.y + offset.y);
		int bottomIndex = computeIndex(sizeX + 1, sizeY + 1, cellStart.x + offset.x, cellStart.y + offset.y - 1);

		//if (cellStart.x == 0 && cellStart.y == 0)
		//{
		//	printf("h[curr]:%f\n", h[currentIndex]);
		//}

		h[currentIndexH] -= dt * ((Fh[currentIndex] - Fh[leftIndex]) * dxi + (Gh[currentIndex] - Gh[bottomIndex]) * dyi);
		hu[currentIndexH] -= dt * ((Fhu[currentIndex] - Fhu[leftIndex]) * dxi + (Ghu[currentIndex] - Ghu[bottomIndex]) * dyi + Bu[currentIndexH] * dxi);
		hv[currentIndexH] -= dt * ((Fhv[currentIndex] - Fhv[leftIndex]) * dxi + (Ghv[currentIndex] - Ghv[bottomIndex]) * dyi + Bv[currentIndexH] * dyi);
	}
	else
	{
		int d = tree[computeIndex(hExt, cellStart + offset)];
		if (d == depth)
		{
			//we don't need to subdivide, we are already at desired accuracy
		}
		else
		{
			//we need to subdivide
			dim3 block(refinementBaseX, refinementBaseY);
			dim3 grid(1, 1);
			eulerTimestep_child_kernel << <grid, block >> >(h, hu, hv,
				Fh, Fhu, Fhv,
				Gh, Ghu, Ghv,
				Bu, Bv,
				tree,
				sizeX, sizeY,
				dt, dxi, dyi,
				refinementBaseX, refinementBaseY, maxRecursions,
				depth + 1, cellStart, cellExt);
		}
	}

}

__global__ void eulerTimestep_kernel(float* h, float* hu, float* hv,
	float* Fh, float* Fhu, float* Fhv,
	float* Gh, float* Ghu, float* Ghv,
	float* Bu, float* Bv,
	int* tree,
	int sizeX, int sizeY,
	float dt, float dxi, float dyi,
	int refinementBaseX, int refinementBaseY, int maxRecursions)
{
	int cellX = threadIdx.x + blockIdx.x * blockDim.x;
	int cellY = threadIdx.y + blockIdx.y * blockDim.y;
	uint2 offset = make_uint2(1, 1);

	uint2 hExt = make_uint2(sizeX + 2, sizeY + 2);
	uint2 computeExt = make_uint2(sizeX, sizeY);
	uint2 cellStart;
	uint2 cellExt;
	computeCellRectangle(computeExt, refinementBaseX, refinementBaseY, maxRecursions, cellX, cellY, cellStart, cellExt);

	//are we out of bounds?
	if (cellStart.x >= computeExt.x || cellStart.y >= computeExt.y)
		return;

	int d = tree[computeIndex(hExt, cellStart + offset)];
	if (d == 0)
	{
		//we don't need to subdivide
		//get the averaged flow on left and right boundary and update cell
	}
	else
	{
		dim3 block(refinementBaseX, refinementBaseY);
		dim3 grid(1, 1);
		eulerTimestep_child_kernel << <grid, block >> >(h, hu, hv,
			Fh, Fhu, Fhv,
			Gh, Ghu, Ghv,
			Bu, Bv,
			tree,
			sizeX, sizeY,
			dt, dxi, dyi,
			refinementBaseX, refinementBaseY, maxRecursions,
			1, cellStart, cellExt);
	}
}

//--------------------------------
//euler timestep
__global__ void getMax_child_kernel(float* h, float* hu, float* hv, int* tree, int sizeX, int sizeY, float2* subOutput, int refinementBaseX, int refinementBaseY, int depth, int maxRecursions, uint2 cellStart, uint2 cellExt)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	uint2 offset = make_uint2(1, 1);
	uint2 ccellStart;
	uint2 ccellExt;
	computeCellRectangle(cellExt, refinementBaseX, refinementBaseY, maxRecursions - depth, i, j, ccellStart, ccellExt);

	if (ccellStart.x >= ccellExt.x || ccellStart.y >= ccellExt.y)
		return;

	//modify cell start to global grid
	cellStart += ccellStart;
	//modify to real cell area
	cellExt = ccellExt;

	int currentIndex = computeIndex(sizeX + 2, sizeY + 2, cellStart.x + offset.x, cellStart.y + offset.y);
	int d = tree[currentIndex];
	if (depth == maxRecursions || d == depth)
	{
		//we can't subdivide anymore
		//or we don't need to subdivide anymore
		float2 maxima = make_float2(h[currentIndex], max(fabsf(hu[currentIndex]), fabsf(hv[currentIndex])));
		subOutput[computeIndex(refinementBaseX, refinementBaseY, i, j)] = maxima;
	}
	else
	{
		//we need to subdivide
		float2* subSubOutput = new float2[refinementBaseX * refinementBaseY];
		dim3 block(refinementBaseX, refinementBaseY);
		dim3 grid(1, 1);
		getMax_child_kernel << <grid, block >> >(h, hu, hv, tree, sizeX, sizeY, subSubOutput, refinementBaseX, refinementBaseY, depth + 1, maxRecursions, cellStart, cellExt);
		cudaDeviceSynchronize();

		//get the maximum subOutput
		float2 maxima = make_float2(0.0f, 0.0f);
		for (int k = 0; k < refinementBaseX * refinementBaseY; k++)
		{
			maxima.x = max(maxima.x, subSubOutput[k].x);
			maxima.y = max(maxima.y, subSubOutput[k].y);
		}

		subOutput[computeIndex(refinementBaseX, refinementBaseY, i, j)] = maxima;
		delete[] subSubOutput;
	}
}

__global__ void getMax_kernel(float* h, float* hu, float* hv, int* tree, int sizeX, int sizeY, float2* output, int refinementBaseX, int refinementBaseY, int maxRecursions)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= sizeX || j >= sizeY)
		return;

	//offset to compute area
	uint2 offset = make_uint2(1, 1);
	uint2 hExt = make_uint2(sizeX + 2, sizeY + 2);
	uint2 hCompute = hExt - make_uint2(2, 2);
	uint2 cellStart;
	uint2 cellExt;
	computeCellRectangle(hCompute, refinementBaseX, refinementBaseY, maxRecursions, i, j, cellStart, cellExt);

	int currentIndex = computeIndex(hExt, cellStart + offset);
	if (tree[currentIndex] == 0)
	{
		//we don't need to subdivide, cell is averaged
		float2 maxima = make_float2(h[currentIndex], max(fabsf(hu[currentIndex]), fabsf(hv[currentIndex])));
		output[computeIndex(sizeX, sizeY, i, j)] = maxima;
	}
	else
	{
		float2* subOutput = new float2[refinementBaseX * refinementBaseY];
		dim3 block(refinementBaseX, refinementBaseY);
		dim3 grid(1, 1);
		getMax_child_kernel << <grid, block >> >(h, hu, hv, tree, sizeX, sizeY, subOutput, refinementBaseX, refinementBaseY, 1, maxRecursions, cellStart, cellExt);
		cudaDeviceSynchronize();

		//get the maximum subOutput
		float2 maxima = make_float2(0.0f, 0.0f);
		for (int k = 0; k < refinementBaseX * refinementBaseY; k++)
		{
			maxima.x = max(maxima.x, subOutput[k].x);
			maxima.y = max(maxima.y, subOutput[k].y);
		}

		output[computeIndex(sizeX, sizeY, i, j)] = maxima;
		delete[] subOutput;
	}
}

//--------------------------------
//averaging of values
//__global__ void getAveragedVerticalValue_child_kernel(float* arr, uint2 globStart, int refinementBase, int myDepth, int maxRecursions, float* resFather, int* numFather)
//{
//	int idxY = threadIdx.x + blockIdx.x * blockDim.x;
//	int baseLengthY = (int)pow(refinementBase, maxRecursions - myDepth);
//}
//
////averaging part of a row / column
//__device__ float getAveragedVerticalValue(float* arr, uint2 start, int refinementBase, int myDepth, int maxRecursions)
//{
//	//check depth of start, if refinement is needed
//	if (arr->getDepths()[computeIndex(arr->getWidth(), arr->getHeight(), start.x, start.y)] == myDepth)
//	{
//		//no subdivision
//		//value at start is already the averaged value
//		return arr->getValues()[computeIndex(arr->getWidth(), arr->getHeight(), start.x, start.y)];
//	}
//	else
//	{
//		//array for sub-averages
//		float* subSums = new float[refinementBase];
//		int* numElems = new int[refinementBase];
//
//		//fill those arrays recursively
//		dim3 block(refinementBase);
//		dim3 grid(1);
//		getAveragedVerticalValue_child_kernel << <grid, block >> >(arr, start, refinementBase, myDepth + 1, subSums, numElems);
//
//		//sum them up
//		float res = 0;
//		int num = 0;
//		for (int i = 0; i < refinementBase; i++)
//		{
//			res += subSums[i];
//			num += numElems[i];
//		}
//
//		delete[] subSums;
//		delete[] numElems;
//
//		return res / num;
//	}
//}
//
//__device__ float getAveragedHorizontalValue(float* arr, uint2 start, int refinementBase, int myDepth)
//{
//
//}
