#include "treeOperations.h"

#define SUBDIV 2
#define THREADS 128

__global__ void testCase(int nx, int ny, int* res)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	TreeElem* gpuTree;
	createFullTree(&gpuTree, nx, ny, SUBDIV);
	cudaDeviceSynchronize();
	res[i] = countLeaves(gpuTree);
	delete gpuTree;
}

int main(int argc, char** argv)
{
	int nx = 16;
	int ny = 16;

	if (argc > 1)
	{
		nx = (int)atoi(argv[1]);
		ny = nx;
	}
	else if (argc > 2)
	{
		nx = (int)atoi(argv[1]);
		ny = (int)atoi(argv[2]);
	}

	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 5);
	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

	size_t lim;
	cudaDeviceGetLimit(&lim, cudaLimitDevRuntimeSyncDepth);
	std::cout << "rsd: " << lim << std::endl;

	lim;
	cudaDeviceGetLimit(&lim, cudaLimitDevRuntimePendingLaunchCount);
	std::cout << "plc: " << lim << std::endl;

	std::cout << "nx: " << nx << " ny: " << ny << std::endl;
	std::cout << "numThreadsX: " << THREADS << " numThreadsY: " << THREADS << std::endl;

	int res[THREADS * THREADS];
	int* res_d;
	checkCudaErrors(cudaMalloc(&res_d, THREADS * THREADS * sizeof(int)));

	dim3 block(16 * 16);
	dim3 grid(THREADS * THREADS / (16 * 16));
	testCase << <grid, block>> >(nx, ny, res_d);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(res, res_d, THREADS * THREADS * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < THREADS * THREADS; i++)
		std::cout << res[i] << std::endl;

	checkCudaErrors(cudaFree(res_d));

	cudaDeviceReset();
	return 0;
}
