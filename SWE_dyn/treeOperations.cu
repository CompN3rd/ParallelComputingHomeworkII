#include "treeOperations.h"

//creation of a full tree
//------------------------------------------------
__global__ void createFullTree_Child(TreeElem** children, int base, int depth, int maxRecursions)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	delete children[i];
	children[i] = new BranchElem(base, base, depth);

	if (depth < maxRecursions - 1)
	{
		dim3 numBlocks(1);
		dim3 numThreads(base * base);

		createFullTree_Child << <numBlocks, numThreads >> >(((BranchElem*)children[i])->children, base, depth + 1, maxRecursions);
	}
}

__global__ void createFullTree_Root(TreeElem** root, int nx, int ny, int base)
{
	int maxRecursions = (int)(ceil(logf(fmaxf(nx, ny))));

	//create the leaf
	if (maxRecursions == 0)
	{
		LeafElem* leaf = new LeafElem();
		root[0] = leaf;
	}
	else
	{
		BranchElem* r = new BranchElem(base, base, 0);

		dim3 numBlocks(1);
		dim3 numThreads(base * base);

		createFullTree_Child << <numBlocks, numThreads >> >(r->children, base, 1, maxRecursions);

		root[0] = r;
	}
}
//------------------------------------------------