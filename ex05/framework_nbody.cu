#include "writeVTK.h"

#ifndef _WIN32
#include <sys/time.h>
double get_time() 
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

#else
#include <Windows.h>
struct Timer
{
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;

	double time = 0.0;
};

void startTimer(Timer& timer)
{
	if (QueryPerformanceFrequency(&timer.frequency) == FALSE)
	{
		fprintf(stderr, "QueryPerformanceFrequency failed \n");
		exit(1);
	}

	if (QueryPerformanceCounter(&timer.start) == FALSE)
	{
		fprintf(stderr, "QueryPerformanceCounter failed \n");
		exit(1);
	}
}

double stopTimer(Timer& timer)
{
	if (QueryPerformanceCounter(&timer.end) == FALSE)
	{
		fprintf(stderr, "QueryPerformanceCounter failed \n");
		exit(1);
	}

	return static_cast<double>(timer.end.QuadPart - timer.start.QuadPart) / timer.frequency.QuadPart;
}
#endif

#define EPS2 0.0001f

int   const N       = 2048;
int   const THREADS = 32;

__inline__ __device__ float length(float3 vec)
{
	return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ float3 bodyBodyInteraction(float4 myPos, float4 otherPos, float3 accel)
{
	float3 r;
	float3 a = accel;

	r.x = otherPos.x - myPos.x;
	r.y = otherPos.y - myPos.y;
	r.z = otherPos.z - myPos.z;

	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
	float distSixth = distSqr * distSqr * distSqr;
	float invDistCube = 1.0f / sqrtf(distSixth);
	float s = otherPos.w * invDistCube;

	a.x += r.x * s;
	a.y += r.y * s;
	a.z += r.z * s;

	return a;
}

__device__ float3 tile_calculation(float4 myPosition, float3 accel)
{
	int i;
	extern __shared__ float4 shPosition[];
#pragma unroll
	for (i = 0; i < blockDim.x; i++)
	{
		accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
	}
	return accel;
}

__global__ void kernel(float4 *sourceGlob, float4 *outGlob, int tileSize) {
	extern __shared__ float4 shPosition[];
	float4 myPosition;

	int i, tile;
	float3 acc = { 0.0f, 0.0f, 0.0f };
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	myPosition = sourceGlob[gtid];
#pragma unroll
	for (i = 0, tile = 0; i < N; i += tileSize, tile++)
	{
		int idx = tile * blockDim.x + threadIdx.x;
		shPosition[threadIdx.x] = sourceGlob[idx];
		__syncthreads();
		acc = tile_calculation(myPosition, acc);
		__syncthreads();
	}
	float4 acc4 = { acc.x, acc.y, acc.z, 0.0f };
	outGlob[gtid] = acc4;
}

__global__ void updatePositions(float4* sourceGlob, float4* accels, float time)
{
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	float4 myPosition = sourceGlob[gtid];
	float4 myAccel = accels[gtid];

	myPosition.x += time * time * myAccel.x;
	myPosition.y += time * time * myAccel.y;
	myPosition.z += time * time * myAccel.z;

	sourceGlob[gtid] = myPosition;
}


int main() {

  int modN = N%THREADS;
  int N1 = N+(THREADS-modN)%THREADS;

  float4 *sourceHost,*sourceDevc, *outDevice, *outHost;
  float  *targetHost,*targetDevc;


  sourceHost = (float4*)     malloc( N1*sizeof(float4) );
  outHost = (float4*)     malloc( N1*sizeof(float4) );
  targetHost = (float*)     malloc( N1*sizeof(float ) );


  cudaMalloc(  (void**) &sourceDevc, N1*sizeof(float4) );
  cudaMalloc(  (void**) &targetDevc, N1*sizeof(float ) );
  cudaMalloc(  (void**) &outDevice, N1*sizeof(float4 ) );
// Initialize
  for( int i=0; i<N; i++ ) {
    sourceHost[i].x = rand()/(1.+RAND_MAX);
    sourceHost[i].y = rand()/(1.+RAND_MAX);
    sourceHost[i].z = rand()/(1.+RAND_MAX);
    sourceHost[i].w = 1.0/N;
  }

  for (int i = N; i<N1; i++) {
    sourceHost[i].x = 0.0f;
    sourceHost[i].y = 0.0f;
    sourceHost[i].z = 0.0f;
    sourceHost[i].w = 0.0f;
  }

  writeVTK("NBody", 0, sourceHost, N);

// Direct summation on device
 
  cudaMemcpy(sourceDevc,sourceHost,N1*sizeof(float4),cudaMemcpyHostToDevice);
  if (modN != 0) printf("Mismatch between N and Blocksize. Modyfing N to match. Old N: %i. New N: %i\n", N, N1);
  printf("Parameters: N: %i, Threads: %i, Blocks: %i\n", N1, THREADS, N1/THREADS);
  double time = 0;

#ifdef _WIN32
  Timer t;
  startTimer(t);
#else
  double  start = get_time();
#endif
  // Aufruf
  for (int i = 0; i < 10; i++) 
  {
      kernel<<< N1/THREADS, THREADS, THREADS * sizeof(float4) >>>(sourceDevc, outDevice, N1/THREADS);
	  updatePositions <<< N1 / THREADS, THREADS >>>(sourceDevc, outDevice, EPS2 * 10.0f);
      cudaMemcpy(outHost,sourceDevc,N1*sizeof(float4) ,cudaMemcpyDeviceToHost);
#ifdef _WIN32
  time += stopTimer(t);
#else
  double  stop = get_time();
  time += stop - start;
  start = get_time();
#endif
  writeVTK("NBody", i+1, outHost, N);
  }

  printf("Time: %f\n", time);

  free(sourceHost);
  free(outHost);
  free(targetHost);

  cudaFree(sourceDevc);
  cudaFree(targetDevc);
  cudaFree(outDevice);
}
