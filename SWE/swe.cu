#include <math.h>
#include "help.hh"
#include "SWE_Block.hh"
#include <sstream>
#include <time.h>
#include <sys/time.h>

/* if macro SPLASH is defined, the "splashing pool" is used as scenario
   -> unset to switch to scenario "radial breaking dam"
 */
// #define SPLASH


/*
 * helper function to check for errors in CUDA calls
 * source: NVIDIA
 */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "\nCuda error (%s): %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}

//-------- Methods to define initial conditions --------

/**
 * function to initialise unknown h (water level):
 * scenario: splashing pool (usewith wall boundaries 
 */
float splash_height(float x, float y) {
  // return 10+(1.0-x);
  // return 10+(1.0-y);
  return 100+(1.0-(x+y));
};

/**
 * function to initialise unknown h (water level):
 * scenario "radial dam break" 
 */ 
float radial_height(float x, float y) {
  return ( sqrt( (x-0.5)*(x-0.5) + (y-0.5)*(y-0.5) ) < 0.1 ) ? 10.1: 10.0;
}

//==================================================================
/**
 * main program for simulation on a single SWE_Block
 * (used, if macro _BLOCK is not defined)
 */
int main(int argc, char** argv) {

  time_t timer1, timer2; /* to calculate execution time */
  clock_t clock1, clock2;
  double cpu_time;

  // define grid size and initial time step
  SWE_Block::initGridData(256,256,0.005,0.005);

  // define Cartesian grid block
  SWE_Block splash;
  
#ifndef SPLASH
  /**
   * init default scenario: radial breaking dam 
   */
  //--- set initial values for simulation:
  // splash.setInitValues(&radial_height, 0.0, 0.0);
  splash.setInitValues(10, 0.0, 0.0);
  // define end of simulated time
  float endSimulation = 0.1;
  
  //--- set initial values for bathymetry:
  splash.setBathymetry(&radial_height);

  // define boundary type at all four domain boundaries:
  splash.setBoundaryType(WALL,WALL,WALL,WALL); // walls at all boundaries
  // set periodic boundaries (overwrites previous settings):
  // splash.connectBoundaries(2,splash, 3); // bottom-top
  // splash.connectBoundaries(0,splash, 1); // left-right

  // define "checkpoints" for visualisation:
  // (=> at each checkpoint in time, an output file is written) 
  int numbChckPts = 9;
  
  SWE_Block::initTimestep(0.005/sqrt(9.81*10.1));

#else
  /**
   * init scenario: water elevation in a pool
   */
  //--- set initial values for simulation:
  splash.setInitValues(&splash_height, 0.0, 0.0);
  // define end of simulated time
  float endSimulation = 0.1;
  
  //--- set initial values for bathymetry:
  splash.setBathymetry(1.0);

  // define boundary type at all four domain boundaries:
  splash.setBoundaryType(WALL,WALL,WALL,WALL); // walls at all boundaries

  // define "checkpoints" for visualisation:
  int numbChckPts = 1;

  SWE_Block::initTimestep(0.005/sqrt(9.81*11));
#endif

  // compute the checkpoints in time
  float* checkPt = new float[numbChckPts+1];
  for(int cp = 0; cp <= numbChckPts; cp++) {
     checkPt[cp] = cp*(endSimulation/numbChckPts);
  };
  
  splash.setBoundaryLayer();
// cout << splash << endl << flush;

  cout << "Write output file: water level at start." << endl;
  
  std::string basename;
  if (argc <= 1) {
     cout << "Please provide filename for output (format: name or /path/name)" << endl;
     cin >> basename;
  } else
     basename = std::string(argv[1]);

  splash.cudaDump();
  // splash.writeVTKFile3D(generateFileName(basename,0));
  splash.writeVTKFile(generateFileName(basename,0));
  // cout << splash;
  
  // time step loop
  cout << "Start simulation ..." << endl;

  timer1 = time(NULL);
  cpu_time = 0.0;

  float t = 0.0;
  // for(int i=1; i<=1; i++) {
  for(int i=1; i<=numbChckPts; i++) {
     clock1 = clock();
     // do time steps until next checkpoint is reached 
     t = splash.simulate(t,checkPt[i]);
     
     // get unknowns from CUDA device
     splash.cudaDump();
     
     clock2 = clock();
     cpu_time += (clock2-clock1)/(double)CLOCKS_PER_SEC;
     cout << "Write output file: water level at time " << t << endl;
     //splash.writeVTKFile3D(generateFileName(basename,i));
     splash.writeVTKFile(generateFileName(basename,i));
  };

  timer2 = time(NULL);
  printf("Execution time: %f sec. %f (CPU sec) \n",(float)(timer2-timer1),cpu_time);
  
  return 0;
}
