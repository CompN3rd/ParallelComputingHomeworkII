#include "SWE_handler.h"

//constructor and destructor
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

	this->blockSize = dim3(bSX, bSY);

	//create managed arrays for grids
	h = new TreeArray(nx + 2, ny + 2);
	hu = new TreeArray(nx + 2, ny + 2);
	hv = new TreeArray(nx + 2, ny + 2);
	b = new TreeArray(nx + 2, ny + 2);

	//and fluxes
	Bu = new TreeArray(nx + 1, ny + 1);
	Bv = new TreeArray(nx + 1, ny + 1);

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

//-------------------------------------------------
//initial values
void SWE_handler::setInitialValues(float h, float u, float v)
{
	//include border
	for (int i = 0; i <= nx + 1; i++)
	{
		for (int j = 0; j <= ny + 1; j++)
		{
			this->h->getValues()[computeIndex(this->h->getWidth(), this->h->getHeight(), i, j)] = h;
			this->hu->getValues()[computeIndex(this->hu->getWidth(), this->hu->getHeight(), i, j)] = h * u;
			this->hv->getValues()[computeIndex(this->hv->getWidth(), this->hv->getHeight(), i, j)] = h * v;

			//set the depths to maximum depth
			this->h->getDepths()[computeIndex(this->h->getWidth(), this->h->getHeight(), i, j)] = this->maxRecursions;
			this->hu->getDepths()[computeIndex(this->hu->getWidth(), this->hu->getHeight(), i, j)] = this->maxRecursions;
			this->hv->getDepths()[computeIndex(this->hv->getWidth(), this->hv->getHeight(), i, j)] = this->maxRecursions;
		}
	}
}

void SWE_handler::setInitialValues(float(*h)(float, float), float u, float v)
{
	//include boundaries
	for (int i = 0; i <= nx + 1; i++)
	{
		for (int j = 0; j <= ny + 1; j++)
		{
			this->h->getValues()[computeIndex(this->h->getWidth(), this->h->getHeight(), i, j)] = h((i-0.5f)*dx, (j-0.5f)*dy);
			this->hu->getValues()[computeIndex(this->hu->getWidth(), this->hu->getHeight(), i, j)] = h((i-0.5f)*dx, (j-0.5f)*dy) * u;
			this->hv->getValues()[computeIndex(this->hv->getWidth(), this->hv->getHeight(), i, j)] = h((i-0.5f)*dx, (j-0.5f)*dy) * v;

			//set the depths to maximum depth
			this->h->getDepths()[computeIndex(this->h->getWidth(), this->h->getHeight(), i, j)] = this->maxRecursions;
			this->hu->getDepths()[computeIndex(this->hu->getWidth(), this->hu->getHeight(), i, j)] = this->maxRecursions;
			this->hv->getDepths()[computeIndex(this->hv->getWidth(), this->hv->getHeight(), i, j)] = this->maxRecursions;
		}
	}
}

//-------------------------------------------------
//bathymetry values
void SWE_handler::setBathymetry(float b)
{
	//include border
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			this->b->getValues()[computeIndex(this->b->getWidth(), this->b->getHeight(), i, j)] = b;
			this->b->getDepths()[computeIndex(this->b->getWidth(), this->b->getHeight(), i, j)] = this->maxRecursions;
		}
	}
	computeBathymetrySources();
}

void SWE_handler::setBathymetry(float(*b)(float, float))
{
	//include border
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			this->b->getValues()[computeIndex(this->b->getWidth(), this->b->getHeight(), i, j)] = b((i - 0.5f)*dx, (j - 0.5f)*dy);
			this->b->getDepths()[computeIndex(this->b->getWidth(), this->b->getHeight(), i, j)] = this->maxRecursions;
		}
	}
	computeBathymetrySources();
}

void SWE_handler::computeBathymetrySources()
{
	//don't use tree refinement here, at least not yet
	dim3 blockDim = this->blockSize;
	dim3 gridDim(divUp(nx + 1, blockSize.x), divUp(ny + 1, blockSize.y));

	computeBathymetrySources_kernel << <gridDim, blockDim >> >(this->h, this->b, this->Bu, this->Bv, this->g, this->maxRecursions);
	checkCudaErrors(cudaGetLastError());
}

//-------------------------------------------------
//boundary computation
void SWE_handler::setBoundaryType(BoundaryType left, BoundaryType right, BoundaryType bottom, BoundaryType top)
{
	this->left = left;
	this->right = right;
	this->top = top;
	this->bottom = bottom;
}

void SWE_handler::setBoundaryLayer()
{
	//top bottom boundary
	dim3 horizontalBlock(blockDim.x * blockDim.x);
	dim3 horizontalGrid(divUp(this->h->getWidth(), horizontalBlock.x));

	setTopBorder_kernel << <horizontalGrid, horizontalBlock >> >(this->h, this->hu, this->hv, this->top);
	setBottomBorder_kernel << <horizontalGrid, horizontalBlock >> >(this->h, this->hu, this->hv, this->bottom);
	checkCudaErrors(cudaGetLastError());

	//left right boundary
	dim3 verticalBlock(blockDim.y * blockDim.y);
	dim3 verticalGrid(divUp(this->h->getHeight(), horizontalBlock.x));

	setRightBorder_kernel << <verticalGrid, verticalBlock >> >(this->h, this->hu, this->hv, this->right);
	setLeftBorder_kernel << <verticalGrid, verticalBlock >> >(this->h, this->hu, this->hv, this->left);
	checkCudaErrors(cudaGetLastError());
}

//-------------------------------------------------
//simulation
float SWE_handler::simulate(float startTime, float endTime)
{
	float t = startTime;

	do
	{
		setBoundaryLayer();

		computeBathymetrySources();

		t += eulerTimestep();

	} while (t < endTime);

	return t;
}

float SWE_handler::eulerTimestep()
{
	float pessimisticFactor = 0.5f;

	computeFluxes();

	//kernel using dynamic parallelism
	eulerTimestep_kernel();

	return pessimisticFactor * dt;
}

//-------------------------------------------------
//fluxes
void SWE_handler::computeFluxes()
{
	dim3 blockDim = this->blockSize;
	dim3 gridDim(divUp(Fh->getWidth(), blockDim.x), divUp(Fh->getHeight(), blockDim.y));
	computeFluxesF_kernel << <gridDim, blockDim >> >(this->h, this->hu, this->hv, this->Fh, this->Fhu, this->Fhv, this->g);

	gridDim = dim3(divUp(Gh->getWidth(), blockDim.x), divUp(Gh->getHeight(), blockDim.y));
	computeFluxesG_kernel << <gridDim, blockDim >> >(this->h, this->hu, this->hv, this->Gh, this->Ghu, this->Ghv, this->g);
}