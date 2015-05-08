#include "Utility.h"

//setting of border values
__global__ void setTopBorder_kernel(float* h, float* hu, float* hv, int sizeX, int sizeY, BoundaryType top);
__global__ void setBottomBorder_kernel(float* h, float* hu, float* hv, int sizeX, int sizeY, BoundaryType bottom);
__global__ void setRightBorder_kernel(float* h, float* hu, float* hv, int sizeX, int sizeY, BoundaryType right);
__global__ void setLeftBorder_kernel(float* h, float* hu, float* hv, int sizeX, int sizeY, BoundaryType left);

//computing the bathymetry
//__global__ void computeHorizontalBathymetryFluxes_kernel(TreeArray* h, TreeArray* b, TreeArray* Bu, float g, int refinementBaseX, int refinementBaseY, int maxRecursions);
__global__ void computeBathymetrySources_kernel(float* h, float* b, float* Bu, float* Bv, int sizeX, int sizeY, float g, int maxRecursions);

//fluxes
__global__ void computeFluxesF_kernel(float* h, float* hu, float* hv, float* Fh, float* Fhu, float* Fhv, int sizeX, int sizeY, float g);
__global__ void computeFluxesG_kernel(float* h, float* hu, float* hv, float* Gh, float* Ghu, float* Ghv, int sizeX, int sizeY, float g);

//euler timestep
__global__ void eulerTimestep_kernel(float* h, float* hu, float* hv,
	float* Fh, float* Fhu, float* Fhv,
	float* Gh, float* Ghu, float* Ghv,
	float* Bu, float* Bv,
	int* tree,
	int sizeX, int sizeY,
	float dt, float dxi, float dyi,
	int refinementBaseX, int refinementBaseY, int maxRecursions);

//maximum of solution vector
__global__ void getMax_kernel(float* h, float* hu, float* hv, int* tree, int sizeX, int sizeY, float2* output, int refinementBaseX, int refinementBaseY, int maxRecursions);

//averaging of values
//__device__ float getAveragedVerticalValue(TreeArray* arr, uint2 start, int refinementBase, int myDepth, int maxRecursions);
//__device__ float getAveragedHorizontalValue(TreeArray* arr, uint2 start, int refinementBase, int myDepth, int maxRecursions);