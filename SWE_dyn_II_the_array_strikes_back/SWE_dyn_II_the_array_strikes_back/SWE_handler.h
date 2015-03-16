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

	//solution
	TreeArray* h;
	TreeArray* hu;
	TreeArray* hv;

	//Bathymetry
	TreeArray* b;

	//fluxes
	TreeArray* Fh;
	TreeArray* Fhu;
	TreeArray* Fhv;
	TreeArray* Gh;
	TreeArray* Ghu;
	TreeArray* Ghv;

	//bathymetry fluxes
	TreeArray* Bu;
	TreeArray* Bv;

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
	float eulerTimestep();
	float getMaxTimestep();

	void computeFluxes();

	inline void setTimestep(float t)
	{
		this->dt = t;
	}
};