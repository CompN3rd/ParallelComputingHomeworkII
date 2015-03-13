#include "SWE_handler.h"
#include <iostream>
using namespace std;

__global__ void makeRect(TreeArray* arr)
{
	dim3 block(16, 16);
	dim3 grid(divUp(arr->nx, 16), divUp(arr->ny, 16));

	fillRect<int> << <grid, block >> >(arr->depths, arr->nx, arr->ny, 0, 0, 0, arr->nx, arr->ny);

	dim3 block2(16, 16);
	dim3 grid2(divUp(arr->nx / 4, 16), divUp(arr->ny / 4, 16));
	fillRect<int> << <grid2, block2 >> >(arr->depths, arr->nx, arr->ny, 5, 0, 0, arr->nx / 4, arr->ny / 4);
}

__global__ void quadTreeRec(int depth, int maxDepth, int* resArr)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (depth == maxDepth)
	{
		resArr[computeIndex(4, 4, i, j)] = 1;
	}
	else
	{
		int* newResArr = new int[4 * 4];
		dim3 block(4, 4);
		dim3 grid(1, 1);

		quadTreeRec << <grid, block >> >(depth + 1, maxDepth, newResArr);
		cudaDeviceSynchronize();

		resArr[computeIndex(4, 4, i, j)] = 0;
		for (int k = 0; k < 4; k++)
			for (int l = 0; l < 4; l++)
				resArr[computeIndex(4, 4, i, j)] += newResArr[computeIndex(4, 4, k, l)];

		delete[] newResArr;
	}
}

__global__ void quadTreeCount(int maxDepth, int* res)
{
	int* resArr = new int[4 * 4];

	dim3 block(4, 4);
	dim3 grid(1, 1);

	quadTreeRec << <grid, block >> >(1, maxDepth, resArr);
	cudaDeviceSynchronize();

	res[0] = 0;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			res[0] += resArr[computeIndex(4, 4, i, j)];

	delete[] resArr;
}

int main()
{
	//TreeArray* arr = new TreeArray(512, 512);

	//makeRect << <1, 1 >> >(arr);
	//checkCudaErrors(cudaDeviceSynchronize());

	//for (int i = 0; i < arr->nx; i++)
	//{
	//	for (int j = 0; j < arr->ny; j++)
	//	{
	//		cout << arr->depths[computeIndex(arr->nx, arr->ny, i, j)] << " ";
	//	}
	//	cout << endl;
	//}

	//delete arr;

	checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 10));

	int* res;
	cudaMallocManaged(&res, sizeof(int));

	quadTreeCount << <1, 1 >> >(4, res);
	checkCudaErrors(cudaDeviceSynchronize());

	cout << res[0] << endl;

	cudaFree(res);

	checkCudaErrors(cudaDeviceReset());
}