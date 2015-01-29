#ifndef __SWE_BLOCK_HH
#define __SWE_BLOCK_HH

#include <iostream>
#include <stdio.h>
#include <fstream>

#include "help.hh"


using namespace std;

// number of blocks for multiple-block-version of the code
const int BLOCKS=4;

// enum type: available types of boundary conditions
typedef enum BoundaryType {
   OUTFLOW, WALL, CONNECT
} BoundaryType;

/**
 * SWE_Block is the main data structure to compute our shallow water model 
 * on a single Cartesian grid block 
 */
class SWE_Block {

  public:
    // Constructor und Destructor
    SWE_Block();
    ~SWE_Block();
    
  // object methods
    // Initialise unknowns:
    void setInitValues(float _h, float _u, float _v);
    void setInitValues(float (*_h)(float, float), float _u, float _v);
    void setBathymetry(float (*_b)(float, float));
    void setBathymetry(float _b);

    // defining boundary conditions
    void setBoundaryType(BoundaryType left, BoundaryType right, 
                         BoundaryType bottom, BoundaryType top);
    void connectBoundaries(int edge, SWE_Block &neighbour, int neighEdge);
    
    // set values in ghost layers (set boundary conditions)
    void setBoundaryLayer();
    // execute Euler time step
    float eulerTimestep();
    // execute Euler time step
    float simulate(float tStart, float tEnd);
    // determine maximum possible time step
    float getMaxTimestep();
    
    // load unknowns h,hu,hv from main memory into CUDA device memory
    void cudaLoadUnknowns();
    // dump unknowns h,hu,hv from CUDA device memory into main memory
    void cudaDump();
    // dump fluxes for h,hu,hv from CUDA device memory into main memory
    void cudaDumpFlux();

    // methods to write a ParaView output file for visualisation: 
    void writeVTKFile(string filename);
    void writeVTKFile3D(string filename);

  // Klassen-Methoden:
    // initialisiere statische Gittervariablen
    static void initGridData(int _nx, int _ny, float _dx, float _dy) {
      nx = _nx; ny = _ny; dx = _dx; dy = _dy; };
    // initialisiere Zeitschrittweite
    static void initTimestep(float _dt) { 
       // cout << " Set new timestep: " << _dt << endl; 
       dt = _dt; 
    };
    
  // Konstanten:
    // Erdbeschleunigung g:
    static const float g;

  private:
     
    // methods to compute flux terms on edges
    void computeFluxes();
    void computeChanges();
    void computeBathymetrySources();


    float computeFlux(float fLoc, float fNeigh, float xiLoc, float xiNeigh, float llf);
    float computeLocalSV(int i, int j, char dir);

    // define arrays for unknowns: 
    // h (water level) and u,v (velocity in x and y direction)
    // hd, ud, and vd are respective CUDA arrays on GPU
    Float2D h; float* hd;
    Float2D hu; float* hud;
    Float2D hv; float* hvd;
    Float2D b; float* bd;
    
    // arrays to hold the values of the flux terms at cell edges
    Float2D Fh; float* Fhd;
    Float2D Fhu; float* Fhud;
    Float2D Fhv; float* Fhvd;
    Float2D Gh; float* Ghd;
    Float2D Ghu; float* Ghud;
    Float2D Ghv; float* Ghvd;
    // arrays to hold the bathymetry source terms for the hu and hv equations
    float* Bxd;
    float* Byd;
    
    // helper arrays: store maximum height and velocities to determine time step
    float* maxhd;
    float* maxvd;
    
    // type of boundary conditions
    BoundaryType boundary[4];
    // for CONNECT boundaries: pointer to connected neighbour block
    SWE_Block* neighbour[4];

  // class variables
  // -> have to be identical for all grid blocks
  
    // grid size: number of cells (incl. ghost layer in x and y direction:
    static int nx;
    static int ny;
    // mesh size dx and dy:
    static float dx;
    static float dy;
    // time step size dt:
    static float dt;

    //--- for visualisation
    // name of output file (for visualisation)
    ofstream Vtk_file;

  //  string FileName;

    // overload operator<< such that data can be written via cout <<
    // -> needs to be declared as friend to be allowed to access private data
    friend ostream& operator<< (ostream& os, const SWE_Block& swe);
    friend void writeOpenmpOutput(std::string FileName, SWE_Block  (&splash)[BLOCKS][BLOCKS]);
    friend void writeVTKContainerFile(string FileName, string basename, int cp);

  
};

ostream& operator<< (ostream& os, const SWE_Block& swe);



#endif
