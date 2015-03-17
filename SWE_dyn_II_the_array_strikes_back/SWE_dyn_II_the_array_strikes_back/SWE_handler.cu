#include "SWE_handler.h"
using namespace std;

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

	//important set synchronization level
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, this->maxRecursions + 1);

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
	dim3 horizontalBlock(this->blockSize.x * this->blockSize.x);
	dim3 horizontalGrid(divUp(this->h->getWidth(), horizontalBlock.x));

	setTopBorder_kernel << <horizontalGrid, horizontalBlock >> >(this->h, this->hu, this->hv, this->top);
	setBottomBorder_kernel << <horizontalGrid, horizontalBlock >> >(this->h, this->hu, this->hv, this->bottom);
	checkCudaErrors(cudaGetLastError());

	//left right boundary
	dim3 verticalBlock(this->blockSize.y * this->blockSize.y);
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
		cout << "currentTime: " << t << endl;

		//get max timestep for this
		//float tMax = getMaxTimestep();
		//this->setTimestep(tMax);

	} while (t < endTime);

	return t;
}

//-------------------------------------------------
//stepping forward in time
float SWE_handler::eulerTimestep()
{
	float pessimisticFactor = 0.5f;

	computeFluxes();

	//kernel using dynamic parallelism
	dim3 block = this->blockSize;
	dim3 grid = dim3(computeForestBase(this->nx, this->refinementBaseX, this->maxRecursions), computeForestBase(this->ny, this->refinementBaseX, this->maxRecursions));

	eulerTimestep_kernel << <grid, block >> >(this->h, this->hu, this->hv,
		this->Fh, this->Fhu, this->Fhv,
		this->Gh, this->Ghu, this->Ghv,
		this->Bu, this->Bv,
		this->dt, this->dx, this->dy,
		this->refinementBaseX, this->refinementBaseY, this->maxRecursions);

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

//-------------------------------------------------
//stepping forward in time
float SWE_handler::getMaxTimestep()
{
	float meshSize = (dx<dy) ? dx : dy;
	float hmax = 0.0f;
	float velmax = 0.0f;
	float2* result;

	dim3 block = this->blockSize;
	dim3 grid(computeForestBase(nx, refinementBaseX, maxRecursions), computeForestBase(ny, refinementBaseY, maxRecursions));

	cudaMallocManaged(&result, grid.x * grid.y * sizeof(float2));

	//to be sure, that the simulation is finished
	checkCudaErrors(cudaDeviceSynchronize());
	getMax_kernel << <grid, block >> >(this->h, this->hu, this->hv, result, grid.x, grid.y, this->refinementBaseX, this->refinementBaseY, this->maxRecursions);
	checkCudaErrors(cudaDeviceSynchronize());

	for (unsigned int i = 0; i < grid.x * grid.y; i++)
	{
		hmax = max(hmax, result[i].x);
		velmax = max(velmax, result[i].y);
	}

	cout << "hmax: " << hmax << " velmax: " << velmax << endl;

	cudaFree(result);

	return meshSize / (sqrtf(this->g * hmax) + velmax);
}

//-------------------------------------------------
//stepping forward in time
void SWE_handler::writeVTKFile(std::string filename)
{
	std::ofstream Vtk_file;
	// VTK HEADER
	Vtk_file.open(filename.c_str());
	Vtk_file << "# vtk DataFile Version 2.0" << endl;
	Vtk_file << "HPC Tutorials: Michael Bader, Kaveh Rahnema, Oliver Meister" << endl;
	Vtk_file << "ASCII" << endl;
	Vtk_file << "DATASET RECTILINEAR_GRID" << endl;
	Vtk_file << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << "1" << endl;
	Vtk_file << "X_COORDINATES " << nx + 1 << " float" << endl;
	//GITTER PUNKTE
	for (int i = 0; i<nx + 1; i++)
		Vtk_file << i*dx << endl;
	Vtk_file << "Y_COORDINATES " << ny + 1 << " float" << endl;
	//GITTER PUNKTE
	for (int i = 0; i<ny + 1; i++)
		Vtk_file << i*dy << endl;
	Vtk_file << "Z_COORDINATES 1 float" << endl;
	Vtk_file << "0" << endl;
	Vtk_file << "CELL_DATA " << ny*nx << endl;
	Vtk_file << "SCALARS H float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	//DOFS
	for (int j = 1; j<ny + 1; j++)
		for (int i = 1; i<nx + 1; i++)
			Vtk_file << (h->getValues()[computeIndex(h->getWidth(), h->getHeight(), i, j)] + b->getValues()[computeIndex(b->getWidth(), b->getHeight(), i, j)]) << endl;
	Vtk_file << "SCALARS U float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j<ny + 1; j++)
		for (int i = 1; i<nx + 1; i++)
			Vtk_file << hu->getValues()[computeIndex(hu->getWidth(), hu->getHeight(), i, j)] / h->getValues()[computeIndex(h->getWidth(), h->getHeight(), i, j)] << endl;
	Vtk_file << "SCALARS V float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j<ny + 1; j++)
		for (int i = 1; i<nx + 1; i++)
			Vtk_file << hv->getValues()[computeIndex(hv->getWidth(), hv->getHeight(), i, j)] / h->getValues()[computeIndex(h->getWidth(), h->getHeight(), i, j)] << endl;
	Vtk_file << "SCALARS B float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j<ny + 1; j++)
		for (int i = 1; i<nx + 1; i++)
			Vtk_file << b->getValues()[computeIndex(b->getWidth(), b->getHeight(), i, j)] << endl;
	Vtk_file.close();
}