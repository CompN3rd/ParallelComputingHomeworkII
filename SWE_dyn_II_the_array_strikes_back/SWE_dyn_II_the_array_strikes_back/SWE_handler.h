#include "Kernels.h"

class SWE_handler
{
private:
	int nx;
	int ny;

	float dx;
	float dy;

	float g;
	float dt;

	//adaptivity of the grid:
	int maxRecursions;
	int refinementBaseX;
	int refinementBaseY;

	dim3 blockSize;

	//grid representation
	int* tree; int* d_tree;

	//solution
	float* h; float* d_h;
	float* hu; float* d_hu;
	float* hv; float* d_hv;

	//Bathymetry
	float* b; float* d_b;

	//fluxes
	float* Fh; float* d_Fh;
	float* Fhu; float* d_Fhu;
	float* Fhv; float* d_Fhv;
	float* Gh; float* d_Gh;
	float* Ghu; float* d_Ghu;
	float* Ghv; float* d_Ghv;

	//bathymetry fluxes
	float* Bu; float* d_Bu;
	float* Bv; float* d_Bv;

	//boundary
	BoundaryType left, top, right, bottom;

public:
	SWE_handler(int nx, int ny, float dx, float dy, float g, int refinementBaseX = 2, int refinementBaseY = 2, int maxRecursions = 2, int blockSizeX = 16, int blockSizeY = 16);
	~SWE_handler();

	void setInitialValues(float h, float u, float v);
	void setInitialValues(float(*h)(float, float), float u, float v);

	void setBathymetry(float b);
	void setBathymetry(float(*b)(float, float));
	void computeBathymetrySources();

	void setBoundaryType(BoundaryType left, BoundaryType right, BoundaryType bottom, BoundaryType top);
	void setBoundaryLayer();

	float simulate(float startTime, float endTime);
	void computeFluxes();
	float eulerTimestep();
	float getMaxTimestep();

	inline void synchronizeSolution()
	{
		checkCudaErrors(cudaMemcpy(this->h, this->d_h, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->hu, this->d_hu, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->hv, this->d_hv, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyDeviceToHost));
	}

	inline void setTimestep(float t)
	{
		this->dt = t;
	}

	//output
	void writeVTKFile(std::string filename);
};