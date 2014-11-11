#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define ZOOM_FACTOR 0.8

// If you want to do some timing stuff.
double get_time()
{  
	struct timeval tim;
	cudaThreadSynchronize();
	gettimeofday(&tim, NULL);
	return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

// Function for writing the picture. Input: Array with with value[i] = 1
// if part of the Julia-set, else 0. Size of the picture. 
void writePPM(float* array_h, int dimX, int dimY) {
	FILE *fp = fopen("picture.ppm", "wb");
	fprintf(fp, "P6\n%d %d\n255\n", dimX, dimY);
	for (int j = 0; j < dimY; ++j)
	{
		for (int i = 0; i < dimX; ++i)
		{
			static unsigned char color[3];
			if (array_h[j*dimX+i] == 1.0f) 
			{
				color[0] = 255;  /* red */
				color[1] = 0;  /* green */
				color[2] = 0;/* blue */
				fwrite(color, 1, 3, fp);
			}
			else 
			{
				color[0] = 0;  /* red */
				color[1] = 0;  /* green */
				color[2] = 0;  /* blue */
				fwrite(color, 1, 3, fp);
			}
		}
	}
	fclose(fp);
	printf("Wrote picture.ppm");
}



// CUDA kernel
__global__ void julia(float* array_d, int dimX, int dimY, int iter)
{

	int myI;
	int myJ;

	float myX;
	float myY;
	float Znr, Zr;
	float Zni, Zi;

	myJ = blockIdx.x;
	myI = blockIdx.y;

	int arrayIndex = blockIdx.y * gridDim.x + blockIdx.x;

	array_d[arrayIndex]=0;

	myX = ZOOM_FACTOR * (2*(float)myJ / gridDim.x - 1);
	myY = ZOOM_FACTOR * (2*(float)myI / gridDim.y - 1);

	if (myX >=-1 && myX<=1 && myY >=-1 && myY<=1)
	{
		Zr = myX;
		Zi = myY;
		for (int it=0; it<iter; it++)
		{
			Znr = Zr*Zr - Zi*Zi - 1.2;
			Zni = 2*Zr*Zi + 0.157;
			Zr = Znr;
			Zi = Zni;
		}

		if (Znr*Znr + Zni*Zni < 1000)
		{
			array_d[arrayIndex] = 1;
		}
	}

}
 

int main( int argc, char* argv[] )
{
	int dimensionX = 1000;
	int dimensionY = 1000;
	int iterations = 300;

	float *array_h, *array_d;

	// ToDo: Allocate Memory on host and device.
	array_h = (float*) malloc(dimensionsX * dimensionsY * sizeof(float));
	cudaMalloc((void**) &array_d, dimensionsX * dimensionsY * sizeof(float));

	// Define the grid, block and thread dimensions. You can either mine, which is a rather simple one
	// or change it like ever you want. You have to think about the indexing in the cuda kernel!
	dim3 dG(dimensionsX, dimensionsY);
	dim3 dB(1);

	// Call the kernel.
	julia<<<dG, dB>>>(array_d, dimensionX, dimensionY, iterations);
  
	//Copy back the results.
	cudaMemcpy(array_h, array_d, sizeof(float)*dimensionX*dimensionY, cudaMemcpyDeviceToHost);

	// Write the picture. Picture will be saved in your home directory!
	writePPM(array_h, dimensionX, dimensionY);

	cudaFree(array_d);
	cudaFreeHost(array_h);

  return 0;
}
