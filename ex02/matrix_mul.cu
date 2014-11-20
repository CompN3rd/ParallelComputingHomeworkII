#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "cuda_utils.h"


// CUDA kernel
/*__global__ void matrix_mul(float* matrixA_d, float* matrixb_d int dimX, int dimY)
{


}*/
 

int main( int argc, char* argv[] )
{
	Timer timer ;
	
	int ToDo = 1;
	int dimensionAx = 2;
	int dimensionAy = 2;
	int dimensionBy = 2;
	int dimensionBx = 2;
	double duration;
	float *matrixA_h, *matrixB_h, *matrixA_d, *matrixB_d, *matrix_erg;
	
	matrixA_h  = (float*) malloc(dimensionAx * dimensionAy * sizeof(float));
	matrixB_h  = (float*) malloc(dimensionBx * dimensionBy * sizeof(float));
	matrix_erg = (float*) malloc(dimensionAy * dimensionBx * sizeof(float)); 
	
	int x,y;
	/*
	 Initialize A and B^T like:
	 |111....	|
	 |222....	|
	 |333		|
	 |.			|
	 |.			|
	 A is stored row wise
	 B is stores column wise 
	 erg is stored row wise
	*/
	for(y=0; y<dimensionAy; y++)
	{
		for(x=0; x<dimensionAx; x++)
		{
			matrixA_h[x+(dimensionAx*y)] = y+1;
		}
	}

	for(x=0; x<dimensionBx; x++)
	{
		for(y=0; y<dimensionBy; y++)
		{
			matrixB_h[y+(dimensionBy*x)] = y+1;
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
	switch(ToDo)
	{
		initTimer (& timer );
		int row,column,columnB;
		case 1: 
			/*Matrix Multiplikation on CPU*/
			for(row=0; row<dimensionAy; row++)
			{
				for(column=0; column<dimensionBx; column++)
				{
					//printf("\nErg[%d]=",(row*dimensionBx)+column);
					matrix_erg[(row*dimensionBx)+column] = 0;						
					for(columnB=0; columnB<dimensionBx; columnB++)
					{
						//printf("+A[%d]*B[%d]", columnB+(dimensionAx*row),columnB+(dimensionBy*column));
						matrix_erg[(row*dimensionBx)+column] += matrixA_h[columnB+(dimensionAx*row)] *matrixB_h[columnB+(dimensionBy*column)];
					}
				}
			}
			for(x=0; x<dimensionAy*dimensionBy; x++)
			{
				printf("\n erg[%d] %f ", x, matrix_erg[x]);
			}
//			cudaThreadSyncronize(); // identifier undefined ???
			duration = getTimer (& timer );
		break;
		case 2: 
			/* Matrix Multiplikation on GPU with device Mem*/
			//TODO:

		
		break;
		case 3: 
			/* Matrix Multiplikation on GPU with shared Mem*/
			//TODO

		
		break;

	}
	
	
	
	
	return 0;
}
