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
	ArrayHelper<int>* tree; int* d_tree;

	//solution
	ArrayHelper<float>* h; float* d_h;
	ArrayHelper<float>* hu; float* d_hu;
	ArrayHelper<float>* hv; float* d_hv;

	//Bathymetry
	ArrayHelper<float>* b; float* d_b;

	//fluxes
	ArrayHelper<float>* Fh; float* d_Fh;
	ArrayHelper<float>* Fhu; float* d_Fhu;
	ArrayHelper<float>* Fhv; float* d_Fhv;
	ArrayHelper<float>* Gh; float* d_Gh;
	ArrayHelper<float>* Ghu; float* d_Ghu;
	ArrayHelper<float>* Ghv; float* d_Ghv;

	//bathymetry fluxes
	ArrayHelper<float>* Bu; float* d_Bu;
	ArrayHelper<float>* Bv; float* d_Bv;

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
		checkCudaErrors(cudaMemcpy(this->h->getValues(), this->d_h, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->hu->getValues(), this->d_hu, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->hv->getValues(), this->d_hv, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyDeviceToHost));
	}

	inline void setTimestep(float t)
	{
		this->dt = t;
	}

	//output
	void writeVTKFile(std::string filename);
};