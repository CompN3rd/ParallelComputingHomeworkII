#include "treeOperations.h"

#define SUBDIV 4
#define THREADS 4

__global__ void testCase(int nx, int ny, int* res)
{
	int i = threadIdx.x;
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
	std::cout << "nx: " << nx << " ny: " << ny << std::endl;
	std::cout << "numThreadsX: " << THREADS << " numThreadsY: " << THREADS << std::endl;

	int res[THREADS * THREADS];
	int* res_d;
	checkCudaErrors(cudaMalloc(&res_d, THREADS * THREADS * sizeof(int)));

	testCase << <1, THREADS * THREADS >> >(nx, ny, res_d);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(res, res_d, THREADS * THREADS * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < THREADS * THREADS; i++)
		std::cout << res[i] << std::endl;

	checkCudaErrors(cudaFree(res_d));

	cudaDeviceReset();
	return 0;
}
