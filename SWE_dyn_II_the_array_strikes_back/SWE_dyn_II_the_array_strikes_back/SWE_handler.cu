#include "SWE_handler.h"

SWE_handler::SWE_handler(int x, int y, float dx, float dy, float g, int rBX, int rBY, int mR, int bSX, int bSY)
{
	this->nx = x;
	this->ny = y;

	this->dx = dx;
	this->dy = dy;

	this->g = g;

	this->refinementBaseX = rBX;
	this->refinementBaseY = rBY;
	this->maxRecursions = mR;

	this->numThreadsX = divUp(this->nx, (unsigned int)pow(refinementBaseX, mR));
	this->numThreadsY = divUp(this->ny, (unsigned int)pow(refinementBaseY, mR));

	this->blockSize = dim3(bSX, bSY);
	this->gridSize = dim3(divUp(this->numThreadsX, this->blockSize.x), divUp(this->numThreadsY, this->blockSize.y));

	//create managed arrays for grids
	h = new TreeArray(nx + 2, ny + 2);
	hu = new TreeArray(nx + 2, ny + 2);
	hv = new TreeArray(nx + 2, ny + 2);
	b = new TreeArray(nx + 2, ny + 2);

	//and fluxes
	Bu = new TreeArray(nx + 2, ny + 2);
	Bv = new TreeArray(nx + 2, ny + 2);

	Fh = new TreeArray(nx + 1, ny + 1);
	Fhu = new TreeArray(nx + 1, ny + 1);
	Fhv = new TreeArray(nx + 1, ny + 1);
	Gh = new TreeArray(nx + 1, ny + 1);
	Ghu = new TreeArray(nx + 1, ny + 1);
	Ghv = new TreeArray(nx + 1, ny + 1);
}

SWE_handler::~SWE_handler()
{
	delete h;
	delete hu;
	delete hv;
	delete b;

	delete Bu;
	delete Bv;

	delete Fh;
	delete Fhu;
	delete Fhv;
	delete Gh;
	delete Ghu;
	delete Ghv;
}