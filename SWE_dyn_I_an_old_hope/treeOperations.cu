#include "treeOperations.h"

//deletion of children
__global__ void deleteChildren(TreeElem** children)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (children[i]->isLeaf())
	{
		delete children[i];
	}
	else
	{
		//recurse over children of children
		BranchElem* r = (BranchElem*)children[i];
		dim3 numBlocks(1);
		dim3 numThreads(r->nx * r->ny);
		deleteChildren << <numBlocks, numThreads >> >(r->children);
		//cudaDeviceSynchronize();
	}
}

//creation of a full tree
//------------------------------------------------
__global__ void createFullTree_Child(TreeElem** children, int base, int depth, int maxRecursions)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (depth < maxRecursions)
	{
		children[i] = new BranchElem(base, base, depth);

		dim3 numBlocks(1);
		dim3 numThreads(base * base);

		createFullTree_Child << <numBlocks, numThreads >> >(((BranchElem*)children[i])->children, base, depth + 1, maxRecursions);
		//cudaDeviceSynchronize();
	}
	else
	{
		children[i] = new LeafElem();
	}
}

__device__ __host__ void createFullTree(TreeElem** root, int nx, int ny, int base)
{
	int maxRecursions = (int)(ceil((logf(fmaxf(nx, ny)))/logf(base)));

	//create the leaf
	if (maxRecursions == 0)
	{
		LeafElem* leaf = new LeafElem();
		root[0] = leaf;
	}
	else
	{
		BranchElem* r = new BranchElem(base, base, 0);
		root[0] = r;

		//create children of root
		dim3 numBlocks(1);
		dim3 numThreads(base * base);

		createFullTree_Child << <numBlocks, numThreads >> >(r->children, base, 1, maxRecursions);
		//cudaDeviceSynchronize();
	}
}
//------------------------------------------------
//counting of leaves
//------------------------------------------------
__global__ void countLeaves_Child(TreeElem** children, int* nc)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (children[i]->isLeaf())
	{
		nc[i] = 1;
	}
	else
	{
		BranchElem* c = (BranchElem*)children[i];
		int numChildren = c->nx * c->ny;

		int* ncNew = new int[numChildren];

		dim3 numBlocks(1);
		dim3 numThreads(numChildren);

		countLeaves_Child << <numBlocks, numThreads >> >(c->children, ncNew);
		//cudaDeviceSynchronize();

		nc[i] = 0;
		for (int j = 0; j < c->nx * c->ny; j++)
		{
			nc[i] += ncNew[j];
		}

		delete[] ncNew;
	}
}

__device__ __host__ int countLeaves(TreeElem* root)
{
	if (root->isLeaf())
	{
		return 1;
	}
	else
	{
		BranchElem* r = (BranchElem*)root;
		int numChildren = r->nx * r->ny;

		int* nc = new int[numChildren];

		dim3 numBlocks(1);
		dim3 numThreads(numChildren);

		countLeaves_Child << <numBlocks, numThreads >> >(r->children, nc);
		//cudaDeviceSynchronize();

		int res = 0;
		for (int i = 0; i < r->nx * r->ny; i++)
		{
			res += nc[i];
		}

		delete[] nc;

		return res;
	}
}