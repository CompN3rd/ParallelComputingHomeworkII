#include "Utility.h"

class SWE_handler
{
private:
	int nx;
	int ny;

	float dx;
	float dy;

	float g;

	//adaptivity of the grid:
	int maxRecursions;
	int refinementBaseX;
	int refinementBaseY;
	//derived from that
	int numThreadsX;
	int numThreadsY;

	dim3 blockSize;
	dim3 gridSize;

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

	//bath fluxes
	TreeArray* Bu;
	TreeArray* Bv;

public:
	SWE_handler(int nx, int ny, float dx, float dy, float g, int refinementBaseX = 2, int refinementBaseY = 2, int maxRecursions = 2, int blockSizeX = 16, int blockSizeY = 16);
	~SWE_handler();
};