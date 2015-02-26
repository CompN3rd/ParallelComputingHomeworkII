#include "SWE_Block.hh"
#include "help.hh"
#include <math.h>

const int TILE_SIZE=8;

/*
 * helper function to read CUDA error codes
 * (implementation in swe.cu */
void checkCUDAError(const char *msg);


// define static variables:
  // block size: numer of cells in x and y direction:
  int SWE_Block::nx = 10;
  int SWE_Block::ny = 10;
  // grid sizes dx and dy:
  float SWE_Block::dx = 0.1;
  float SWE_Block::dy = 0.1;
  // time step size dt:
  float SWE_Block::dt = 0.01;

  // gravitational acceleration
  const float SWE_Block::g = 9.81;

/**
 * Constructor: allocate variables for simulation
 *
 * unknowns h,hu,hv,b are defined on grid indices [0,..,nx+1]*[0,..,ny+1]
 * -> computational domain is [1,..,nx]*[1,..,ny]
 * -> plus ghost cell layer
 *
 * flux terms are defined for edges with indices [0,..,nx]*[1,..,ny]
 * or [1,..,nx]*[0,..,ny] (for horizontal/vertical edges)
 * Flux term with index (i,j) is located on the edge between 
 * cells with index (i,j) and (i+1,j) or (i,j+1)
 *
 * bathymetry source terms are defined for cells with indices [1,..,nx]*[1,..,ny]
 */
SWE_Block::SWE_Block() : h(nx+2,ny+2), hu(nx+2,ny+2), hv(nx+2,ny+2),
                         Fh(nx+1,ny+1), Fhu(nx+1,ny+1), Fhv(nx+1,ny+1),
                         Gh(nx+1,ny+1), Ghu(nx+1,ny+1), Ghv(nx+1,ny+1),
			 b(nx+2,ny+2)
{
  // set WALL as default boundary condition
  for(int i=0; i<4;i++) {
     boundary[i] = WALL;
     neighbour[i] = this;
  };
  

  // TODO allocate CUDA memory for unknows h,hu,hv and bathymetry b
     cudaMalloc(&hd,sizeof(float)*(nx+2)*(ny+2));   // Checked
     cudaMalloc(&hud,sizeof(float)*(nx+2)*(ny+2));  // Checked
     cudaMalloc(&hvd,sizeof(float)*(nx+2)*(ny+2));  // Checked
     cudaMalloc(&bd,sizeof(float)*(nx+2)*(ny+2));   // Checked
  // TODO allocate CUDA memory for flow unknows in x directino
     cudaMalloc(&Fhd,sizeof(float)*(nx+1)*(ny+1));  // Checked
     cudaMalloc(&Fhud,sizeof(float)*(nx+1)*(ny+1)); 
     cudaMalloc(&Fhvd,sizeof(float)*(nx+1)*(ny+1));      
  // TODO allocate CUDA memory for flow unknows in y direction
     cudaMalloc(&Ghd,sizeof(float)*(nx+1)*(ny+1));  
     cudaMalloc(&Ghud,sizeof(float)*(nx+1)*(ny+1)); 
     cudaMalloc(&Ghvd,sizeof(float)*(nx+1)*(ny+1)); 
  // TODO allocate CUDA unknowns: bathymetry source terms in x and y
     cudaMalloc(&Bxd,sizeof(float)*(nx+2)*(ny+2)); 
     cudaMalloc(&Byd,sizeof(float)*(nx+2)*(ny+2));
 }

/**
 * Destructor: de-allocate all variables
 */
SWE_Block::~SWE_Block() {
  cudaFree(hd); cudaFree(hud); cudaFree(hvd); cudaFree(bd);
  cudaFree(Fhd); cudaFree(Fhud); cudaFree(Fhvd);
  cudaFree(Ghd); cudaFree(Ghud); cudaFree(Ghvd);
  cudaFree(Bxd); cudaFree(Byd);
  
}

/**
 * perform forward-Euler time steps, starting with simulation time tStart,:
 * until simulation time tEnd is reached; 
 * intended as main simulation loop between two checkpoints
 */
__host__
float SWE_Block::simulate(float tStart, float tEnd) {
  float t = tStart;
  do {

     // set values in ghost cells:
     setBoundaryLayer();
     
     //transfer data to the device
     cudaLoadUnknowns();
     // compute bathymetry source terms (depend on h)
     computeBathymetrySources();

     // execute Euler time step:
     t += eulerTimestep();
     // load the data from the device
     cudaDump();
     // cudaDumpFlux();

     cout << "Simulation at time " << t << endl << flush;

     // calculate and set largest allowed time step:
     float tMax = getMaxTimestep();
     initTimestep(tMax);
  } while(t < tEnd);

  return t;
}

/**
 * Initialise unknowns in all grid cells to a uniform value
 * Note: unknowns hu and hv represent momentum, while parameters u and v are velocities! 
 */
void SWE_Block::setInitValues(float _h, float _u, float _v) {

  //for(int i=1; i<=nx; i++)
  //  for(int j=1; j<=ny; j++) {
	for (int i = 0; i <= nx + 1; i++)
	{
		for (int j = 0; j <= ny + 1; j++) 
		{
			h[i][j] = _h;
			hu[i][j] = _h * _u;
			hv[i][j] = _h * _v;
		}
	}

  cudaLoadUnknowns();
}

/**
 * Initialise unknowns hu and hv in all grid cells to a uniform value;
 * water level is initialised via the specified function _h
 * Note: unknowns hu and hv represent momentum, while parameters u and v are velocities! 
 */
void SWE_Block::setInitValues(float (*_h)(float, float), float _u, float _v) {

  //for(int i=1; i<=nx; i++)
  //  for(int j=1; j<=ny; j++) {
	for (int i = 0; i <= nx + 1; i++)
	{
		for (int j = 0; j <= ny + 1; j++) 
		{
			h[i][j] = _h((i - 0.5)*dx, (j - 0.5)*dy);
			hu[i][j] = _u * h[i][j];
			hv[i][j] = _v * h[i][j];
		}
	}

  cudaLoadUnknowns();
}

/**
 * Initialise bathymetry b in all grid cells (incl. ghost/boundary layers)
 * using a given bathymetry function;
 * bathymetry source terms are re-computed
 */
void SWE_Block::setBathymetry(float (*_b)(float, float)) {

  for(int i=0; i<=nx+1; i++)
    for(int j=0; j<=ny+1; j++)
      b[i][j] = _b((i-0.5)*dx,(j-0.5)*dy);

  cout << "Load bathymetry unknowns into device memory" << flush << endl;
  int size = (nx+2)*(ny+2)*sizeof(float);
  // TODO copy bathymetry into device memory
  cudaMemcpy(bd,b.elemVector(),size,cudaMemcpyHostToDevice);

  //compute the source terms for the first computation
  computeBathymetrySources();
}

/**
 * Initialise bathymetry b in all grid cells (incl. ghost/boundary layers)
 * to a uniform value
 * bathymetry source terms are re-computed
 */
void SWE_Block::setBathymetry(float _b) {

  for(int i=0; i<=nx+1; i++)
    for(int j=0; j<=ny+1; j++)
      b[i][j] = _b;

  cout << "Load bathymetry unknowns into device memory" << flush << endl;
  int size = (nx+2)*(ny+2)*sizeof(float);
  
	// TODO copy bathymetry into device memory
  cudaMemcpy(bd,b.elemVector(),size,cudaMemcpyHostToDevice);
  computeBathymetrySources();
}

/**
 * set boundary type for the four block boundaries
 */
void SWE_Block::setBoundaryType(BoundaryType left, BoundaryType right, 
                                BoundaryType bottom, BoundaryType top) {
  boundary[0] = left;
  boundary[1] = right;
  boundary[2] = bottom;
  boundary[3] = top;
}

/**
 * define a CONNECT boundary:
 * the block boundary with index egde (see method setBoundaryType()) 
 * is connect to the boundary neighEdge of block neighBlock
 */
void SWE_Block::connectBoundaries(int edge, SWE_Block &neighBlock, int neighEdge) {

  boundary[edge] = CONNECT;
  neighBlock.boundary[neighEdge] = CONNECT;

  neighbour[edge] = &neighBlock;
  neighBlock.neighbour[neighEdge] = this;
}

///**
// * CUDA kernel to set left/right boundary layer
// * threadIdx-x == 0 will set left boundary
// * threadIdx.x == 1 will set right boundary
// * blockIdx.y and threadIdy.y loop over the boundary elements
// */
//__global__
//void kernelLeftRightBoundary(float* hd, float* hud, float* hvd,
//                             int nx, int ny, BoundaryType left, BoundaryType right)
//{
//  int i = threadIdx.x*(nx+1);
//  int j = 1 + TILE_SIZE*blockIdx.y + threadIdx.y;
//  int ghost = i*(nx+2) + j;
//  int inner = ( threadIdx.x == 0 ) ? (nx+2) + j : nx*(nx+2) + j;
//  
//  BoundaryType bound = ( threadIdx.x == 0 ) ? left : right;

//  hd[ghost] = hd[inner];
//  hud[ghost] = (bound==WALL) ? -hud[inner] : hud[inner];
//  hvd[ghost] = hvd[inner]; 
//}

///**
// * CUDA kernel to set top/bottom boundary layer
// * threadIdx.y == 0 will set bottom boundary
// * threadIdx.Y == 1 will set top boundary
// * blockIdx.x and threadIdy.x loop over the boundary elements
// */
//__global__
//void kernelBottomTopBoundary(float* hd, float* hud, float* hvd,
//                             int nx, int ny, BoundaryType bottom, BoundaryType top)
//{
//  int i = 1 + TILE_SIZE*blockIdx.x + threadIdx.x;
//  int j = threadIdx.y*(ny+1);
//  int ghost = i*(nx+2) + j;
//  int inner = ( threadIdx.x == 0 ) ? i*(nx+2) + 1 : i*(nx+2) + ny;
//  
//  BoundaryType bound = ( threadIdx.y == 0 ) ? bottom : top;

//  hd[ghost] = hd[inner];
//  hud[ghost] = hud[inner];
//  hvd[ghost] = (bound==WALL) ? -hvd[inner] : hvd[inner]; 
//}

///**
// * set the values of all ghost cells depending on the specifed 
// * boundary conditions
// */
//void SWE_Block::setBoundaryLayer() {

//cout << "Call kernel to compute left/right boundaries " << flush << endl;
//  dim3 dimLeftBlock(2,TILE_SIZE);
//  dim3 dimLeftGrid(1,ny/TILE_SIZE);
//  kernelLeftRightBoundary<<<dimLeftGrid,dimLeftBlock>>>(hd,hud,hvd,
//                                                        nx,ny,
//							boundary[0],boundary[1]);

//cout << "Call kernel to compute bottom/top boundaries " << flush << endl;
//  dim3 dimTopBlock(TILE_SIZE,2);
//  dim3 dimTopGrid(nx/TILE_SIZE,1);
//  kernelBottomTopBoundary<<<dimTopGrid,dimTopBlock>>>(hd,hud,hvd,
//                                                      nx,ny,
//					              boundary[2],boundary[3]);
//}

void SWE_Block::setBoundaryLayer() {

  // Left Boundary
  if (boundary[0]==CONNECT)
     for(int j=1; j<=ny; j++) {
       h[0][j] = neighbour[0]->h[nx][j];
       hu[0][j] = neighbour[0]->hu[nx][j];
       hv[0][j] = neighbour[0]->hv[nx][j]; 
     }  
  else 
     for(int j=1; j<=ny; j++) {
       h[0][j] = h[1][j];
       hu[0][j] = (boundary[0]==WALL) ? -hu[1][j] : hu[1][j];
       hv[0][j] = hv[1][j]; 
     };

  // Right Boundary
  if (boundary[1]==CONNECT)
     for(int j=1; j<=ny; j++) {
       h[nx+1][j] = neighbour[1]->h[1][j];
       hu[nx+1][j] = neighbour[1]->hu[1][j];
       hv[nx+1][j] = neighbour[1]->hv[1][j]; 
     }  
  else 
     for(int j=1; j<=ny; j++) {
       h[nx+1][j] = h[nx][j];
       hu[nx+1][j] = (boundary[1]==WALL) ? -hu[nx][j] : hu[nx][j];
       hv[nx+1][j] = hv[nx][j]; 
     };

  // Bottom Boundary
  if (boundary[2]==CONNECT)
     for(int i=1; i<=nx; i++) {
       h[i][0] = neighbour[2]->h[i][ny];
       hu[i][0] = neighbour[2]->hu[i][ny];
       hv[i][0] = neighbour[2]->hv[i][ny]; 
     }  
  else 
     for(int i=1; i<=nx; i++) {
       h[i][0] = h[i][1];
       hu[i][0] = hu[i][1];
       hv[i][0] = (boundary[2]==WALL) ? -hv[i][1] : hv[i][1]; 
     };

  // Top Boundary
  if (boundary[3]==CONNECT)
     for(int i=1; i<=nx; i++) {
       h[i][ny+1] = neighbour[3]->h[i][1];
       hu[i][ny+1] = neighbour[3]->hu[i][1];
       hv[i][ny+1] = neighbour[3]->hv[i][1]; 
     }  
  else 
     for(int i=1; i<=nx; i++) {
       h[i][ny+1] = h[i][ny];
       hu[i][ny+1] = hu[i][ny];
       hv[i][ny+1] = (boundary[3]==WALL) ? -hv[i][ny] : hv[i][ny]; 
     };

  // only required for visualisation: set values in corner ghost cells
  h[0][0] = h[1][1];
  h[0][ny+1] = h[1][ny];
  h[nx+1][0] = h[nx][1];
  h[nx+1][ny+1] = h[nx][ny];
}

float SWE_Block::getMaxTimestep() {

  float hmax = 0.0;
  float vmax = 0.0;
  float meshSize = (dx<dy) ? dx : dy;
   
  for(int i=1; i<=nx; i++)
    for(int j=1; j<=ny; j++) {
      if (h[i][j]> hmax) hmax = h[i][j];
      if (fabs(hu[i][j])> vmax) vmax = fabs(hu[i][j]);
      if (fabs(hv[i][j])> vmax) vmax = fabs(hv[i][j]);
    };

  cout << "hmax " << hmax << endl << flush;
  cout << "vmax " << vmax << endl << flush;

  // sqrt(g*hmax) + vmax is the velocity of a characteristic shallow-water wave
  // such that a wave must not propagate farther than dx in a single time step
  return meshSize/(sqrt(g*hmax) + vmax);
}


/**
 * CUDA kernel for Euler time step
 */
__global__
void kernelEulerTimestep(float* hd, float* hud, float* hvd,
                         float* Fhd, float* Fhud, float* Fhvd,
                         float* Ghd, float* Ghud, float* Ghvd,
			 float* Bxd, float* Byd,
                         int nx, int ny, float dt, float dxi, float dyi)
{
// TODO implement that kernel
// it should compute the new values of hd, hud and hvd

     int i= blockIdx.x*blockDim.x+threadIdx.x+1;
   int j= blockIdx.y*blockDim.y+threadIdx.y+1;

   if (i>0 && i<=nx && j>0 && j<=ny)
   {
	    hd [i* (nx+2) +j] -= dt*((Fhd [i* (nx+1) +j]-Fhd [(i-1)* (nx+1) +j])*dxi + (Ghd [i* (nx+1) +j]-Ghd [i* (nx+1) +j-1])*dyi );
	    hud[i* (nx+2) +j] -= dt*((Fhud[i* (nx+1) +j]-Fhud[(i-1)* (nx+1) +j])*dxi + (Ghud[i* (nx+1) +j]-Ghud[i* (nx+1) +j-1])*dyi + Bxd[i*(nx+2)+j]*dxi);
	    hvd[i* (nx+2) +j] -= dt*((Fhvd[i* (nx+1) +j]-Fhvd[(i-1)* (nx+1) +j])*dxi + (Ghvd[i* (nx+1) +j]-Ghvd[i* (nx+1) +j-1])*dyi +Byd[i*(nx+2)+j]*dyi);
   }
}

/**
 * execute a single forward-Euler time step:
 * depending on the flux terms in Fh, Gh, etc., compute the balance terms  
 * for each cell, and update the unknowns accordingly.
 * return value is the time step.
 */

__host__ 
float SWE_Block::eulerTimestep() 
{
	float pes = 0.5; // pessimistic factor to decrease time step size

	//combute the fluxes on the device
	computeFluxes();

	//TODO set up the cuda dimBlock and dimGrid with the appropriate sizes


	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((nx - 1) / TILE_SIZE + 1, (ny - 1) / TILE_SIZE + 1);

	cout << "Call kernel for Euler timestep " << flush << endl;

	//call the kernel for the computation of a Eulertimestep
	kernelEulerTimestep <<< dimGrid, dimBlock >>> (hd, hud, hvd,
		Fhd, Fhud, Fhvd, Ghd, Ghud, Ghvd,
		Bxd, Byd,
		nx, ny, pes*dt, 1.0f / dx, 1.0f / dy);

	//Original c-code
	/*
		for(int i=1; i<=nx; i++)
		for(int j=1; j<=ny; j++) {
		h[i][j] -= pes*dt *( (Fh[i][j]-Fh[i-1][j])/dx + (Gh[i][j]-Gh[i][j-1])/dy );
		u[i][j] -= pes*dt *( (Fu[i][j]-Fu[i-1][j])/dx + (Gu[i][j]-Gu[i][j-1])/dy );
		v[i][j] -= pes*dt *( (Fv[i][j]-Fv[i-1][j])/dx + (Gv[i][j]-Gv[i][j-1])/dy );
		};
		*/

	return pes*dt;
}

/**
 * compute the flux term on a given edge 
 * (acc. to local Lax-Friedrich method aka Rusanov flux):
 * fLow and fHigh contain the values of the flux function in the two 
 * adjacent grid cells 
 * xiLow and xiHigh are the values of the unknowns in the two 
 * adjacent grid cells 
 * "Low" represents the cell with lower i/j index ("High" for larger index). 
 * llf should contain the local signal velocity (as compute by computeLocalSV)
 * for llf=dx/dt (or dy/dt), we obtain the standard Lax Friedrich method
 */
float SWE_Block::computeFlux(float fLow, float fHigh, float xiLow, float xiHigh,
                              float llf) {
  // local Lax-Friedrich
  return 0.5*(fLow+fHigh) - 0.5*llf*(xiHigh-xiLow);
}

/* Same function but can only be executed on the device */
__device__
float computeFlux(float fLow, float fHigh, float xiLow, float xiHigh, float llf) {
  // local Lax-Friedrich
  return 0.5f*(fLow+fHigh) - 0.5f*llf*(xiHigh-xiLow);
}

__global__
void kernelComputeFluxesF(float* hd, float* hud, float* hvd,
                          float* Fhd, float* Fhud, float* Fhvd,
                          int nx, float g,int istart)
{
//TODO Implement that kernel for computing the fluxes over edges. istart is the index of the first column for which the flux is computed on the righthand edge. use the devices' computeFlux method
   
   // WARNING: The algorithm comes from WE_Block::computeFluxes().
//Original c-code
/*
  // fluxes in x direction:
  for(int i=0; i<=nx; i++)
    for(int j=1; j<=ny; j++) {
      double llf = computeLocalSV(i,j,'x');
      // h-Konponente
      Fh[i][j] = computeFlux( h[i][j]*u[i][j], h[i+1][j]*u[i+1][j], h[i][j], h[i+1][j], llf );
      Fu[i][j] = computeFlux( u[i][j]*u[i][j] + 0.5*g*h[i][j],
                              u[i+1][j]*u[i+1][j] + 0.5*g*h[i+1][j],
			      u[i][j], 
			      u[i+1][j], llf );
      Fv[i][j] = computeFlux( u[i][j]*v[i][j],u[i+1][j]*v[i+1][j], v[i][j], v[i+1][j], llf );
    };
  */

   int i= blockIdx.x*blockDim.x+threadIdx.x+istart;
   int j= blockIdx.y*blockDim.y+threadIdx.y+1;
   float sv1, sv2,llf;

   if (i<=nx && j <= nx )
   {
      //float llf = computeLocalSV(i,j,'x');
      sv1 = fabs(hud[i* (nx+2) +j]     /hd[i* (nx+2) +j])     + sqrt(g*hd[i*     (nx+2) +j]);
      sv2 = fabs(hud[(i+1)* (nx+2) +j] /hd[(i+1)* (nx+2) +j]) + sqrt(g*hd[(i+1)* (nx+2) +j]);
      llf = (sv1 > sv2) ? sv1 : sv2;

      Fhd[i* (nx+1) +j] = computeFlux  (
                  hd[i*     (nx+2) +j]*hud[i*     (nx+2) +j], 
                  hd[(i+1)* (nx+2) +j]*hud[(i+1)* (nx+2) +j],
                  hd[i*     (nx+2) +j],
                  hd[(i+1)* (nx+2) +j],
                  llf);

      Fhud[i* (nx+1) +j] = computeFlux (
                  hud[i*     (nx+2) +j] *hud[i*     (nx+2) +j]+ (0.5*g*hd[i*     (nx+2) +j]),
                  hud[(i+1)* (nx+2) +j] *hud[(i+1)* (nx+2) +j]+ (0.5*g*hd[(i+1)* (nx+2) +j]),
                  hud[i*     (nx+2) +j],
                  hud[(i+1)* (nx+2) +j],
                  llf);

      Fhvd[i* (nx+1) +j] = computeFlux (
                  hud[i*     (nx+2) +j]* hvd[i*     (nx+2) +j],
                  hud[(i+1)* (nx+2) +j]* hvd[(i+1)* (nx+2) +j],
                  hvd[i*     (nx+2) +j],
                  hvd[(i+1)* (nx+2) +j],
                  llf);
   }

 }

__global__
void kernelComputeFluxesG(float* hd, float* hud, float* hvd,
                          float* Ghd, float* Ghud, float* Ghvd,
                          int nx, float g, int jstart)
{
//TODO Implement that kernel for computing the fluxes over edges. jstart is the index of the first row for which the flux is computed on the lower edge
// fluxes in y direction:
  /*for(int j=0; j<=ny; j++)
    for(int i=1; i<=nx; i++) {
      double llf = computeLocalSV(i,j,'y');
      // h-Konponente
      Gh[i][j] = computeFlux( h[i][j]*v[i][j], h[i][j+1]*v[i][j+1], h[i][j], h[i][j+1], llf );
      Gu[i][j] = computeFlux( u[i][j]*v[i][j],u[i][j+1]*v[i][j+1], u[i][j], u[i][j+1], llf );
      Gv[i][j] = computeFlux( v[i][j]*v[i][j] + 0.5*g*h[i][j],
                              v[i][j+1]*v[i][j+1] + 0.5*g*h[i][j+1],
			      v[i][j], 
			      v[i][j+1], llf );
    };*/

   int i= blockIdx.x*blockDim.x+threadIdx.x+1;
   int j= blockIdx.y*blockDim.y+threadIdx.y+jstart;
   float sv1,sv2,llf;

   if (i<=nx && j<=nx )
   {
      //float llf = computeLocalSV(i,j,'y');
      sv1 = fabs(hvd[i* (nx+2) +j]   /hd[i* (nx+2) +j])   + sqrt(g*hd[i* (nx+2) +j]);
      sv2 = fabs(hvd[i* (nx+2) +j+1] /hd[i* (nx+2) +j+1]) + sqrt(g*hd[i* (nx+2) +j+1]);
      llf = (sv1 > sv2) ? sv1 : sv2;

      Ghd[i*  (nx+1) +j] = computeFlux (
                  hd[i* (nx+2) +j]  * hvd[i* (nx+2) +j], 
                  hd[i* (nx+2) +j+1]* hvd[i* (nx+2) +j+1],
                  hd[i* (nx+2) +j],
                  hd[i* (nx+2) +j+1],
                  llf);

      Ghud[i* (nx+1) +j] = computeFlux (
                  hud[i* (nx+2) +j]  * hvd[i* (nx+2) +j],
                  hud[i* (nx+2) +j+1]* hvd[i* (nx+2) +j+1],
                  hud[i* (nx+2) +j],
                  hud[i* (nx+2) +j+1],
                  llf);

      Ghvd[i* (nx+1) +j] = computeFlux (
                  hvd[i* (nx+2) +j]  * hvd[i* (nx+2) +j]  + (0.5*g*hd[i* (nx+2) +j]),
                  hvd[i* (nx+2) +j+1]* hvd[i* (nx+2) +j+1]+ (0.5*g*hd[i* (nx+2) +j+1]),
                  hvd[i* (nx+2) +j],
                  hvd[i* (nx+2) +j+1],
                  llf);

   }
}

/**
 * compute the flux terms on all edges
 */
void SWE_Block::computeFluxes() {

   cout << "Call kernel to compute F-fluxes " << flush << endl;
//TODO set up grids and blocks for the computation of the fluxes. Be aware, that n+1 edges have to be computed, thus the computation has to be divided into a kernel which acts on n data and another one which only acts on a single edge

   dim3 dimBlock1(TILE_SIZE,TILE_SIZE);
   dim3 dimBlock2(1,TILE_SIZE);
   //dim3 dimGrid(nx/TILE_SIZE+1,(ny-1)/TILE_SIZE+1);
   dim3 dimGrid1(nx/TILE_SIZE,ny/TILE_SIZE);
   dim3 dimGrid2(1,ny/TILE_SIZE);


//TODO call the kernel two times with the different grids and blocks

  int istart =0;
  kernelComputeFluxesF<<<dimGrid1,dimBlock1>>>(hd,hud,hvd,Fhd,Fhud,Fhvd,nx,g,istart); 
  istart =nx; 
  kernelComputeFluxesF<<<dimGrid2,dimBlock2>>>(hd,hud,hvd,Fhd,Fhud,Fhvd,nx,g,istart); 
  
  cout << "Call kernel to compute G-fluxes " << flush << endl;

//TODO set up grids and blocks for the computation of the fluxes. Be aware, that n+1 edges have to be computed, thus the computation has to be divided into a kernel which acts on n data and another one which only acts on a single edge

  dim3 dimBlock3(TILE_SIZE,1);
  //dim3 dimGrid((nx-1)/TILE_SIZE+1,ny/TILE_SIZE+1);
  dim3 dimGrid3(nx/TILE_SIZE,1);

//TODO call the kernel two times with the different grids and blocks
  int jstart =0;
  kernelComputeFluxesG<<<dimGrid1,dimBlock1>>>(hd,hud,hvd,Ghd,Ghud,Ghvd,nx,g,jstart);  
  jstart =ny;
  kernelComputeFluxesG<<<dimGrid3,dimBlock3>>>(hd,hud,hvd,Ghd,Ghud,Ghvd,nx,g,jstart);  
		
}


/**
 * computes the local signal velocity in x- or y-direction
 * for two adjacent cells with indices (i,j) and (i+1,j)
 * (if dir='x') or (i,j+1) (if dir='y'
 */
float SWE_Block::computeLocalSV(int i, int j, char dir) {
  float sv1, sv2;
  
  if (dir=='x') {
     sv1 = fabs(hu[i][j]/h[i][j]) + sqrt(g*h[i][j]);
     sv2 = fabs(hu[i+1][j]/h[i+1][j]) + sqrt(g*h[i+1][j]);
  } else {
     sv1 = fabs(hv[i][j]/h[i][j]) + sqrt(g*h[i][j]);
     sv2 = fabs(hv[i][j+1]/h[i][j+1]) + sqrt(g*h[i][j+1]);
  };
  
  return (sv1 > sv2) ? sv1 : sv2;
}

__global__
void kernelComputeBathymetrySources(float* hd, float* bd, float* Bxd, float* Byd, 
                                    int nx, float g)
{
//TODO implement that kernel for computing the bathymetric sources, i.e. all terms related to earths gravity

   int i= blockIdx.x*blockDim.x+threadIdx.x;
   int j= blockIdx.y*blockDim.y+threadIdx.y;

   if (i<=nx+1  && j<=nx+1)
   {
      Bxd[i* (nx+2) +j] = g*(hd[i*(nx+2)+j] * bd[i*(nx+2)+j]-hd[(i-1)*(nx+2)+j] * bd[(i-1)*(nx+2)+j]);
      Byd[i* (nx+2) +j] = g*(hd[i*(nx+2)+j] * bd[i*(nx+2)+j]-hd[i*(nx+2)+j-1] * bd[i*(nx+2)+j-1]);
   }
}

/**
 * compute the bathymetry source terms in all cells
 */
void SWE_Block::computeBathymetrySources() {

  dim3 dimBlock(TILE_SIZE,TILE_SIZE);
  dim3 dimGrid((nx+1)/TILE_SIZE+1,(ny+1)/TILE_SIZE+1);

cout << "Call kernel to compute bathymetry sources" << flush << endl;

  kernelComputeBathymetrySources<<<dimGrid,dimBlock>>>(hd,bd,Bxd,Byd,nx,g);

// cout << (*this) << flush;

}

//==================================================================
// member functions: transfer between main and CUDA device memory
//==================================================================

/** 
  * load unknowns h,hu,hv from main memory into CUDA device memory,
  */
void SWE_Block::cudaLoadUnknowns() {
//TODO load the vectors h, hu, hv into the device memory elements hd, hud, hvd
    
  cout << "Load unknowns into device memory" << flush << endl;
  int size = (nx+2)*(ny+2)*sizeof(float);
  cudaMemcpy(hd,h.elemVector(),size,cudaMemcpyHostToDevice);
  cudaMemcpy(hud,hu.elemVector(),size,cudaMemcpyHostToDevice);
  cudaMemcpy(hvd,hv.elemVector(),size,cudaMemcpyHostToDevice);
}

/** 
  * dump unknowns h,hu,hv from CUDA device memory into main memory
  */
void SWE_Block::cudaDump() {
//TODO load the vectors hd, hud, hvd into the host memory elements hd, hud, hvd
  int size = (nx+2)*(ny+2)*sizeof(float);
  cout << "Copy unknowns from device" << flush << endl;
  cudaMemcpy(h.elemVector(),hd,size,cudaMemcpyDeviceToHost);
  cudaMemcpy(hu.elemVector(),hud,size,cudaMemcpyDeviceToHost);
  cudaMemcpy(hv.elemVector(),hvd,size,cudaMemcpyDeviceToHost);

  // only required for visualisation: set values in corner ghost cells
  h[0][0] = h[1][1];
  h[0][ny+1] = h[1][ny];
  h[nx+1][0] = h[nx][1];
  h[nx+1][ny+1] = h[nx][ny];

}

/** 
  * dump fluxes from CUDA device memory into main memory
  */
void SWE_Block::cudaDumpFlux() {
  // WARNING: This is not right, in the constructor Fh, Gh, etc were defined to be of size (nx+1)*(ny+1)
  // int size = (nx+2)*(ny+2)*sizeof(float);
  int size = (nx+1)*(ny+1)*sizeof(float);

//TODO copy the flux data Fh,Fhu,Fhv, Gh, Ghu , Ghv from the device to the host
  cudaMemcpy(Fh.elemVector(),Fhd,size,cudaMemcpyDeviceToHost);
  cudaMemcpy(Fhu.elemVector(),Fhud,size,cudaMemcpyDeviceToHost);
  cudaMemcpy(Fhv.elemVector(),Fhvd,size,cudaMemcpyDeviceToHost);
  cudaMemcpy(Gh.elemVector(),Ghd,size,cudaMemcpyDeviceToHost);
  cudaMemcpy(Ghu.elemVector(),Ghud,size,cudaMemcpyDeviceToHost);
  cudaMemcpy(Ghv.elemVector(),Ghvd,size,cudaMemcpyDeviceToHost);
}

//==================================================================
// member functions for visualisation and output
//==================================================================


/**
 * Write a VTK file for visualisation using ParaView
 * -> writes h, u, and v unknowns of a single SWE_Block
 *    as a RECTILINEAR grid for ParaView
 */
void SWE_Block::writeVTKFile(string FileName) {

	// VTK HEADER
	Vtk_file.open(FileName.c_str());
	Vtk_file <<"# vtk DataFile Version 2.0"<<endl;
	Vtk_file << "HPC Tutorials: Michael Bader, Kaveh Rahnema, Oliver Meister"<<endl;
	Vtk_file << "ASCII"<<endl;
	Vtk_file << "DATASET RECTILINEAR_GRID"<<endl;
	Vtk_file << "DIMENSIONS "<< nx+1<<" "<<ny+1<<" "<<"1"<<endl;
	Vtk_file <<"X_COORDINATES "<< nx+1 <<" float"<<endl;
	//GITTER PUNKTE
	for (int i=0;i<nx+1;i++)
				Vtk_file << i*dx<<endl;
	Vtk_file <<"Y_COORDINATES "<< ny+1 <<" float"<<endl;
	//GITTER PUNKTE
	for (int i=0;i<ny+1;i++)
				Vtk_file << i*dy<<endl;
	Vtk_file <<"Z_COORDINATES 1 float"<<endl;
	Vtk_file <<"0"<<endl;
	Vtk_file << "CELL_DATA "<<ny*nx<<endl;
	Vtk_file << "SCALARS H float 1"<<endl;
	Vtk_file << "LOOKUP_TABLE default"<<endl;
	//DOFS
	for (int j=1; j<ny+1;j++)
		for (int i=1;i<nx+1;i++)
			Vtk_file <<(h[i][j]+b[i][j])<<endl;
	Vtk_file << "SCALARS U float 1"<<endl;
	Vtk_file << "LOOKUP_TABLE default"<<endl;
	for (int j=1; j<ny+1;j++)	
		for (int i=1;i<nx+1;i++)
			Vtk_file <<hu[i][j]/h[i][j]<<endl;
	Vtk_file << "SCALARS V float 1"<<endl;
	Vtk_file << "LOOKUP_TABLE default"<<endl;
	for (int j=1; j<ny+1;j++)	
		for (int i=1;i<nx+1;i++)
			Vtk_file <<hv[i][j]/h[i][j]<<endl;
	Vtk_file << "SCALARS B float 1"<<endl;
	Vtk_file << "LOOKUP_TABLE default"<<endl;
	for (int j=1; j<ny+1;j++)	
		for (int i=1;i<nx+1;i++)
			Vtk_file <<b[i][j]<<endl;
	Vtk_file.close();

}

/**
 * Write a VTK file for visualisation using ParaView
 * -> writes h, u, and v unknowns of a single SWE_Block
 *    as a STRUCTURED grid for ParaView
 *    (allows 3D effect for water surface)
 */
void SWE_Block::writeVTKFile3D(string FileName) {

	// VTK HEADER
	Vtk_file.open(FileName.c_str());
	Vtk_file <<"# vtk DataFile Version 2.0"<<endl;
	Vtk_file << "HPC Tutorials: Michael Bader, Kaveh Rahnema, Oliver Meister"<<endl;
	Vtk_file << "ASCII"<<endl;
	Vtk_file << "DATASET STRUCTURED_GRID"<<endl;
	Vtk_file << "DIMENSIONS "<< nx+1<<" "<<ny+1<<" "<<"1"<<endl;
	Vtk_file << "POINTS "<<(nx+1)*(ny+1)<<" float"<<endl;
	//GITTER PUNKTE
	for (int j=0; j<ny+1;j++)
			for (int i=0;i<nx+1;i++)
				Vtk_file << i*dx<<" "<<j*dy<<" "
				         << 0.25*(h[i][j]+h[i+1][j]+h[i][j+1]+h[i+1][j+1]
					         +b[i][j]+b[i+1][j]+b[i][j+1]+b[i+1][j+1]) 
					 <<endl;
	Vtk_file <<endl;
	Vtk_file << "CELL_DATA "<<ny*nx<<endl;
	Vtk_file << "SCALARS H float 1"<<endl;
	Vtk_file << "LOOKUP_TABLE default"<<endl;
	//DOFS
	for (int j=1; j<ny+1;j++)
		for (int i=1;i<nx+1;i++)
			Vtk_file <<(h[i][j]+b[i][j])<<endl;
	Vtk_file << "SCALARS U float 1"<<endl;
	Vtk_file << "LOOKUP_TABLE default"<<endl;
	for (int j=1; j<ny+1;j++)	
		for (int i=1;i<nx+1;i++)
			Vtk_file <<hu[i][j]/h[i][j]<<endl;
	Vtk_file << "SCALARS V float 1"<<endl;
	Vtk_file << "LOOKUP_TABLE default"<<endl;
	for (int j=1; j<ny+1;j++)	
		for (int i=1;i<nx+1;i++)
			Vtk_file <<hv[i][j]/h[i][j]<<endl;
	Vtk_file << "SCALARS B float 1"<<endl;
	Vtk_file << "LOOKUP_TABLE default"<<endl;
	for (int j=1; j<ny+1;j++)	
		for (int i=1;i<nx+1;i++)
			Vtk_file <<b[i][j]<<endl;
	Vtk_file.close();

}


/*
 * end of single patch output function
 */

//==================================================================

/**
 * overload operator<< such that data can be written via cout <<
 * -> needs to be declared as friend to be allowed to access private data
 */
ostream& operator<<(ostream& os, const SWE_Block& swe) {
  
  os << "Gitterzellen: " << swe.nx << "x" << swe.ny << endl;

  cout << "Wellenhoehe:" << endl;
  for(int i=0; i<=swe.nx+1; i++) {
    for(int j=0; j<=swe.ny+1; j++) {
      os << swe.h[i][j] << "  ";
    };
    os << endl;
  };

  cout << "Geschwindigkeit in x-Richtung:" << endl;
  for(int i=0; i<=swe.nx+1; i++) {
    for(int j=0; j<=swe.ny+1; j++) {
      os << swe.hu[i][j] << "  ";
    };
    os << endl;
  };

  cout << "Geschwindigkeit in y-Richtung:" << endl;
  for(int i=0; i<=swe.nx-1; i++) {
    for(int j=0; j<=swe.ny-1; j++) {
      os << swe.hv[i][j] << "  ";
    };
    os << endl;
  };

  cout << "Fluss - Wellenhoehe:" << endl;
  for(int i=0; i<=swe.nx; i++) {
    for(int j=1; j<=swe.ny; j++) {
      os << swe.Fh[i][j] << "  ";
    };
    os << endl;
  };

  cout << "Fluss - Durchfluss in x-Richtung:" << endl;
  for(int i=0; i<=swe.nx; i++) {
    for(int j=1; j<=swe.ny; j++) {
      os << swe.Fhu[i][j] << "  ";
    };
    os << endl;
  };

  cout << "Fluss - Durchfluss in y-Richtung:" << endl;
  for(int i=0; i<=swe.nx; i++) {
    for(int j=1; j<=swe.ny; j++) {
      os << swe.Fhv[i][j] << "  ";
    };
    os << endl;
  };

  cout << "Fluss - Wellenhoehe:" << endl;
  for(int i=1; i<=swe.nx; i++) {
    for(int j=0; j<=swe.ny; j++) {
      os << swe.Gh[i][j] << "  ";
    };
    os << endl;
  };

  cout << "Fluss - Durchfluss in x-Richtung:" << endl;
  for(int i=1; i<=swe.nx; i++) {
    for(int j=0; j<=swe.ny; j++) {
      os << swe.Ghu[i][j] << "  ";
    };
    os << endl;
  };

  cout << "Fluss - Durchfluss in y-Richtung:" << endl;
  for(int i=1; i<=swe.nx; i++) {
    for(int j=0; j<=swe.ny; j++) {
      os << swe.Ghv[i][j] << "  ";
    };
    os << endl;
  };
  
  os << flush;

  return os;
}


