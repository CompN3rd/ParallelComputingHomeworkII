#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "cuda_utils.h"

__constant__ int TILE_WIDTH;

// Shared Memory implementation
 __global__ void mat_mul(float *a, float *b, float *ab, int width)
{
	 //non square blocks not supported
	 if (blockDim.x != blockDim.y)
		 return;

	// define tiles in accordance to blockDims
	 int TX = blockDim.x;
	 int TY = blockDim.y;

	// shorthand
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;
	// allocate tiles in __shared__ memory
	__shared__ float s_a[TY][TX];
	__shared__ float s_b[TY][TX];
	// calculate the row & col index
	int row = by*blockDim.y + ty;
	int col = bx*blockDim.x + tx;
	float result = 0;
	// loop over the tiles of the input in phases
	for(int p = 0; p < width/TX; ++p)
	{
		// collaboratively load tiles into __shared__
		s_a[ty][tx] = a[row*width + (p*TX + tx)];
		s_b[ty][tx] = b[(p*TY + ty)*width + col];
		__syncthreads();
		// dot product between row of s_a and col of s_b
		for(int k = 0; k < TX; ++k)
			result += s_a[ty][k] * s_b[k][tx];
		__syncthreads();
	}
	ab[row*width+col] = result;
}

// naive global memory implementation
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
	// Calculate the row index of the Pd element and M
	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	// Calculate the column index of Pd and N
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
	float Pvalue = 0;
	// each thread computes one element of the block sub-matrix
	
	for (int k = 0; k < Width; ++k)
	{
		// printf("\n Row/Col %d/%d Pvalue += A[%d] * B[%d]",Row,Col,Row*Width+k,k*Width+Col);
		Pvalue += Md[Row*Width+k] * Nd[k * Width + Col];
	}
	Pd[Row*Width+Col] = Pvalue;
}

int main( int argc, char* argv[] )
{
	Timer timer ;
	double duration;	
	
	int ToDo = 2;
	
	float *matrixA_h, *matrixB_h, *matrixA_d, *matrixB_d, *matrix_erg, *matrix_erg_d;
		
	int dimensionAx = 2;
	int dimensionAy = 2;
	int dimensionBy = 2;
	int dimensionBx = 2;
	
	int sizeA = dimensionAx * dimensionAy * sizeof(float);
	int sizeB = dimensionBx * dimensionBy * sizeof(float);
	
	matrixA_h  = (float*) malloc(sizeA);
	matrixB_h  = (float*) malloc(sizeB);
	matrix_erg = (float*) malloc(dimensionAy * dimensionBx * sizeof(float)); 

	cudaMalloc(&matrixA_d, sizeA);
	cudaMalloc(&matrixB_d, sizeB);
	cudaMalloc(&matrix_erg_d,dimensionAx*dimensionBy*sizeof(float));
	
	int dimG = 2;				
	int dimBX = (dimensionAx/dimG);
	int dimBY = (dimensionBy/dimG);

	cudaMemcpyToSymbol(&TILE_WIDTH,&dimBX,sizeof(int),0,cudaMemcpyHostToDevice);
	
	dim3 gridDim(dimG, dimG);
	dim3 blockDim(dimBX, dimBY);
	
	int x,y;
	/*
	 Initialize A and B like:
	 |1    2    3....	|
	 |w+1  w+2....	|
	 |2w+2 2w+2...		|
	 |.			|
	 |.			|
	 A is stored row wise
	 B is stored row wise 
	 erg is stored row wise
	*/
	int counter = 0;
	for(y=0; y<dimensionAy; y++)
	{
		for(x=0; x<dimensionAx; x++)
		{
			matrixA_h[x+(dimensionAx*y)] = ++counter;
		}
	}

	counter = 0;
	for(x=0; x<dimensionBx; x++)
	{
		for(y=0; y<dimensionBy; y++)
		{
			matrixB_h[x+(dimensionBx*y)] = ++counter;
		}
	}
	/* // Print the Matrix
	printf("\nA:\n");
	for(x=0 ; x < dimensionAx*dimensionAy; x++)
	{
		printf(" %f", matrixA_h[x]);
	}
	printf("\nB:\n");
	for(x=0 ; x < dimensionBx*dimensionBy; x++)
	{
		printf(" %f", matrixB_h[x]);
	}
	
	printf("\n nach dem initialisieren \n");
	*/
	for(ToDo = 1; ToDo <= 3; ToDo++)
	{
		switch(ToDo)
		{
			case 1:
				initTimer (&timer);
				int row,column,k;
				/*********************************
 				*	Matrix Multiplikation on CPU * 
				*********************************/
				for(row=0; row<dimensionAy; row++)
				{
					for(column=0; column<dimensionBx; column++)
					{
						//printf("\nErg[%d]=",(row*dimensionBx)+column);
						matrix_erg[(row*dimensionBx)+column] = 0;						
						for(k=0; k < dimensionAx; k++)
						{
							//printf("+A[%d]*B[%d]", columnB+(dimensionAx*row),columnB+(dimensionBy*column));
							matrix_erg[(row*dimensionBx)+column] += matrixA_h[(row * dimensionAx) + k] *matrixB_h[(k * dimensionBx) + column];
						}
					}
				}
				duration = getTimer(&timer);

				for(x=0; x<dimensionAy*dimensionBy; x++)
				{
					printf("\n erg[%d] %f ", x, matrix_erg[x]);
				}
				printf("\n duration: %f", duration);
			break;
			case 2: 
			/* Matrix Multiplikation on GPU with device Mem*/
			/****************************************************
 			 *	Matrix Multiplikation on GPU without shared Mem * 
			 ****************************************************/
			 // 
					initTimer(&timer);
					cudaMemcpy(matrixA_d,matrixA_h,sizeA, cudaMemcpyHostToDevice);
					cudaMemcpy(matrixB_d,matrixB_h,sizeB, cudaMemcpyHostToDevice);
				
					/* need: 
					*	maximal thread per block number
					*  maximal number of block which run at the same time
					*
					*/
					// {dimensionX | dimensionX € N , dimensionX % 2 = 0} 
					// {dimensionY | dimensionY € N , dimensionY % 2 = 0} 

					MatrixMulKernel<<<gridDim, blockDim>>>(matrixA_d,matrixB_d,matrix_erg_d,dimensionAx);

					cudaMemcpy(matrix_erg, matrix_erg_d,dimensionAx*dimensionBy*sizeof(float) , cudaMemcpyDeviceToHost);
					cudaThreadSynchronize();
					duration = getTimer(&timer);

					for(x=0; x<dimensionAy*dimensionBy; x++)
					{
						printf("\n erg[%d] %f ", x, matrix_erg[x]);
					}
					printf("\n duration: %f", duration);
			break;

			case 3: 
				/* Matrix Multiplication on GPU with shared Mem*/
				initTimer (&timer);
				cudaMemcpy(matrixA_d,matrixA_h,sizeA, cudaMemcpyHostToDevice);
				cudaMemcpy(matrixB_d,matrixB_h,sizeB, cudaMemcpyHostToDevice);
				
				mat_mul <<<gridDim, blockDim >>> (matrixA_d, matrixB_d, matrix_erg_d, dimensionAx);

				cudaMemcpy(matrix_erg, matrix_erg_d,dimensionAx*dimensionBy*sizeof(float) , cudaMemcpyDeviceToHost);
				cudaThreadSynchronize();

				duration = getTimer(&timer);
				for(x=0; x<dimensionAy*dimensionBy; x++)
				{
					printf("\n erg[%d] %f ", x, matrix_erg[x]);
				}
				printf("\n duration: %f", duration);

			break;
		}
	}
	
	cudaFree(matrixA_d);
	cudaFree(matrixB_d);
	cudaFree(matrix_erg_d);
	free(matrixA_h);
	free(matrixB_h);
	free(matrix_erg);
	
	return 0;
}
