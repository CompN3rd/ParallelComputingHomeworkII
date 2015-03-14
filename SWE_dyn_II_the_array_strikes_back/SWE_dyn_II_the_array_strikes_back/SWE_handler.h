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
	SWE_handler(int x, int y, float dx, float dy, float g, int refinementBaseX, int refinementBaseY, int maxRecursions);
	~SWE_handler();
};