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

	refinementBaseX = rBX;
	refinementBaseY = rBY;
	maxRecursions = mR;

	//important set synchronization level
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, maxRecursions + 1);

	blockSize = dim3(bSX, bSY);

	//create array for adaptivity:
	tree = new int[(nx + 2) * (ny + 2)];
	checkCudaErrors(cudaMalloc(&d_tree, (nx + 2) * (ny + 2) * sizeof(int)));

	//create arrays for grids
	h = new float[(nx + 2) * (ny + 2)];
	checkCudaErrors(cudaMalloc(&d_h, (nx + 2) * (ny + 2) * sizeof(float)));
	hu = new float[(nx + 2) * (ny + 2)];
	checkCudaErrors(cudaMalloc(&d_hu, (nx + 2) * (ny + 2) * sizeof(float)));
	hv = new float[(nx + 2) * (ny + 2)];
	checkCudaErrors(cudaMalloc(&d_hv, (nx + 2) * (ny + 2) * sizeof(float)));
	b = new float[(nx + 2) * (ny + 2)];
	checkCudaErrors(cudaMalloc(&d_b, (nx + 2) * (ny + 2) * sizeof(float)));

	Bu = new float[(nx + 2) * (ny + 2)];
	checkCudaErrors(cudaMalloc(&d_Bu, (nx + 2) * (ny + 2) * sizeof(float)));
	Bv = new float[(nx + 2) * (ny + 2)];
	checkCudaErrors(cudaMalloc(&d_Bv, (nx + 2) * (ny + 2) * sizeof(float)));

	//and fluxes
	Fh = new float[(nx + 1) * (ny + 1)];
	checkCudaErrors(cudaMalloc(&d_Fh, (nx + 1) * (ny + 1) * sizeof(float)));
	Fhu = new float[(nx + 1) * (ny + 1)];
	checkCudaErrors(cudaMalloc(&d_Fhu, (nx + 1) * (ny + 1) * sizeof(float)));
	Fhv = new float[(nx + 1) * (ny + 1)];
	checkCudaErrors(cudaMalloc(&d_Fhv, (nx + 1) * (ny + 1) * sizeof(float)));
	Gh = new float[(nx + 1) * (ny + 1)];
	checkCudaErrors(cudaMalloc(&d_Gh, (nx + 1) * (ny + 1) * sizeof(float)));
	Ghu = new float[(nx + 1) * (ny + 1)];
	checkCudaErrors(cudaMalloc(&d_Ghu, (nx + 1) * (ny + 1) * sizeof(float)));
	Ghv = new float[(nx + 1) * (ny + 1)];
	checkCudaErrors(cudaMalloc(&d_Ghv, (nx + 1) * (ny + 1) * sizeof(float)));
}

SWE_handler::~SWE_handler()
{
	delete[] h;
	checkCudaErrors(cudaFree(d_h));
	delete[] hu;
	checkCudaErrors(cudaFree(d_hu));
	delete[] hv;
	checkCudaErrors(cudaFree(d_hv));
	delete[] b;
	checkCudaErrors(cudaFree(d_b));

	delete[] Bu;
	checkCudaErrors(cudaFree(d_Bu));
	delete[] Bv;
	checkCudaErrors(cudaFree(d_Bv));

	delete[] Fh;
	checkCudaErrors(cudaFree(d_Fh));
	delete[] Fhu;
	checkCudaErrors(cudaFree(d_Fhu));
	delete[] Fhv;
	checkCudaErrors(cudaFree(d_Fhv));
	delete[] Gh;
	checkCudaErrors(cudaFree(d_Gh));
	delete[] Ghu;
	checkCudaErrors(cudaFree(d_Ghu));
	delete[] Ghv;
	checkCudaErrors(cudaFree(d_Ghv));
}

//-------------------------------------------------
//initial values
void SWE_handler::setInitialValues(float h, float u, float v)
{
	//include border
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			this->h[computeIndex(nx + 2, ny + 2, i, j)] = h;
			this->hu[computeIndex(nx + 2, ny + 2, i, j)] = h * u;
			this->hv[computeIndex(nx + 2, ny + 2, i, j)] = h * v;

			//set the depths to maximum depth
			tree[computeIndex(nx + 2, ny + 2, i, j)] = maxRecursions;
		}
	}
	checkCudaErrors(cudaMemcpy(d_h, this->h, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hu, this->hu, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hv, this->hv, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tree, tree, (nx + 2) * (ny + 2) * sizeof(int), cudaMemcpyHostToDevice));
}

void SWE_handler::setInitialValues(float(*h)(float, float), float u, float v)
{
	//include boundaries
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			this->h[computeIndex(nx + 2, ny + 2, i, j)] = h((i-0.5f)*dx, (j-0.5f)*dy);
			this->hu[computeIndex(nx + 2, ny + 2, i, j)] = h((i-0.5f)*dx, (j-0.5f)*dy) * u;
			this->hv[computeIndex(nx + 2, ny + 2, i, j)] = h((i-0.5f)*dx, (j-0.5f)*dy) * v;

			//set the depths to maximum depth
			tree[computeIndex(nx + 2, ny + 2, i, j)] = maxRecursions;
		}
	}
	checkCudaErrors(cudaMemcpy(d_h, this->h, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hu, this->hu, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hv, this->hv, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tree, tree, (nx + 2) * (ny + 2) * sizeof(int), cudaMemcpyHostToDevice));
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
			this->b[computeIndex(nx + 2, ny + 2, i, j)] = b;
		}
	}
	checkCudaErrors(cudaMemcpy(d_b, this->b, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	computeBathymetrySources();
}

void SWE_handler::setBathymetry(float(*b)(float, float))
{
	//include border
	for (int i = 0; i < nx + 2; i++)
	{
		for (int j = 0; j < ny + 2; j++)
		{
			this->b[computeIndex(nx + 2, ny + 2, i, j)] = b((i - 0.5f)*dx, (j - 0.5f)*dy);
		}
	}
	checkCudaErrors(cudaMemcpy(d_b, this->b, (nx + 2) * (ny + 2) * sizeof(float), cudaMemcpyHostToDevice));
	computeBathymetrySources();
}

void SWE_handler::computeBathymetrySources()
{
	//don't use tree refinement here, at least not yet
	dim3 blockDim = blockSize;
	dim3 gridDim(divUp(nx + 1, blockSize.x), divUp(ny + 1, blockSize.y));

	computeBathymetrySources_kernel << <gridDim, blockDim >> >(d_h, d_b, d_Bu, d_Bv, nx, ny, g, maxRecursions);
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
	dim3 horizontalBlock(blockSize.x * blockSize.x);
	dim3 horizontalGrid(divUp(nx + 2, horizontalBlock.x));

	setTopBorder_kernel << <horizontalGrid, horizontalBlock >> >(d_h, d_hu, d_hv, nx, ny, top);
	setBottomBorder_kernel << <horizontalGrid, horizontalBlock >> >(d_h, d_hu, d_hv, nx, ny, bottom);
	checkCudaErrors(cudaGetLastError());

	//left right boundary
	dim3 verticalBlock(blockSize.y * blockSize.y);
	dim3 verticalGrid(divUp(ny + 2, horizontalBlock.x));

	setRightBorder_kernel << <verticalGrid, verticalBlock >> >(d_h, d_hu, d_hv, nx, ny, right);
	setLeftBorder_kernel << <verticalGrid, verticalBlock >> >(d_h, d_hu, d_hv, nx, ny, left);
	checkCudaErrors(cudaGetLastError());
}

//-------------------------------------------------
//simulation
float SWE_handler::simulate(float startTime, float endTime)
{
	float t = startTime;

	do
	{
		//set values in ghost cells:
		setBoundaryLayer();

		//compute bathymetry source terms
		computeBathymetrySources();

		//execute Euler time step
		t += eulerTimestep();

		//get max timestep for this
		float tMax = getMaxTimestep();
		setTimestep(tMax);

	} while (t < endTime);

	return t;
}

//-------------------------------------------------
//stepping forward in time
float SWE_handler::eulerTimestep()
{
	float pessimisticFactor = 0.5f;

	//compute the fluxes on the device
	computeFluxes();

	//kernel using dynamic parallelism
	dim3 block = blockSize;
	dim3 grid = dim3(computeForestBase(nx, refinementBaseX, maxRecursions), computeForestBase(ny, refinementBaseY, maxRecursions));

	eulerTimestep_kernel << <grid, block >> >(d_h, d_hu, d_hv,
		d_Fh, d_Fhu, d_Fhv,
		d_Gh, d_Ghu, d_Ghv,
		d_Bu, d_Bv,
		d_tree,
		nx, ny,
		pessimisticFactor *dt, 1.0f / dx, 1.0f / dy,
		refinementBaseX, refinementBaseY, maxRecursions);

	return pessimisticFactor * dt;
}

//-------------------------------------------------
//fluxes
void SWE_handler::computeFluxes()
{
	dim3 blockDim = blockSize;
	dim3 gridDim(divUp(nx + 1, blockDim.x), divUp(ny + 1, blockDim.y));

	computeFluxesF_kernel << <gridDim, blockDim >> >(d_h, d_hu, d_hv, d_Fh, d_Fhu, d_Fhv, nx, ny, g);
	computeFluxesG_kernel << <gridDim, blockDim >> >(d_h, d_hu, d_hv, d_Gh, d_Ghu, d_Ghv, nx, ny, g);
}

//-------------------------------------------------
//stepping forward in time
float SWE_handler::getMaxTimestep()
{
	float meshSize = (dx<dy) ? dx : dy;
	float hmax = 0.0f;
	float velmax = 0.0f;
	float2* result;
	float2* d_result;

	dim3 block = blockSize;
	dim3 grid(computeForestBase(nx, refinementBaseX, maxRecursions), computeForestBase(ny, refinementBaseY, maxRecursions));

	result = new float2[grid.x * grid.y];
	checkCudaErrors(cudaMalloc(&d_result, grid.x * grid.y * sizeof(float2)));

	//to be sure, that the simulation is finished
	getMax_kernel << <grid, block >> >(d_h, d_hu, d_hv, d_tree, grid.x, grid.y, d_result, refinementBaseX, refinementBaseY, maxRecursions);
	checkCudaErrors(cudaMemcpy(result, d_result, grid.x * grid.y * sizeof(float2), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < grid.x * grid.y; i++)
	{
		hmax = max(hmax, fabsf(result[i].x));
		velmax = max(velmax, fabsf(result[i].y));
	}

	cout << "hmax: " << hmax << " velmax: " << velmax << endl;

	delete[] result;
	cudaFree(d_result);

	return meshSize / (sqrtf(g * hmax) + velmax);
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
			Vtk_file << (h[computeIndex(nx + 2, ny + 2, i, j)] + b[computeIndex(nx + 2, ny + 2, i, j)]) << endl;
	Vtk_file << "SCALARS U float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j<ny + 1; j++)
		for (int i = 1; i<nx + 1; i++)
			Vtk_file << hu[computeIndex(nx + 2, ny + 2, i, j)] / h[computeIndex(nx + 2, ny + 2, i, j)] << endl;
	Vtk_file << "SCALARS V float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j<ny + 1; j++)
		for (int i = 1; i<nx + 1; i++)
			Vtk_file << hv[computeIndex(nx + 2, ny + 2, i, j)] / h[computeIndex(nx + 2, ny + 2, i, j)] << endl;
	Vtk_file << "SCALARS B float 1" << endl;
	Vtk_file << "LOOKUP_TABLE default" << endl;
	for (int j = 1; j<ny + 1; j++)
		for (int i = 1; i<nx + 1; i++)
			Vtk_file << b[computeIndex(nx + 2, ny + 2, i, j)] << endl;
	Vtk_file.close();
}