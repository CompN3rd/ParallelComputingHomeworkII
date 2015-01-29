#ifndef __HELP_HH
#define __HELP_HH


#include <iostream>
#include <fstream>
#include <sstream>

/**
 * class Float2D is a very basic helper class to deal with 2D float arrays:
 * indices represent columns (1st index) and rows (2nd index) of a 2D grid;
 * values are sequentially ordered in memory (column by column).
 * Besides constructor/deconstructor, the class provides overloading of 
 * the []-operator, such that elements can be accessed as a[i][j]. 
 */ 
class Float2D
{
  public:
    Float2D(int _rows, int _cols) : rows(_rows),cols(_cols)
    {
      elem = new float[rows*cols];
      for(int i = 0;i < rows; i++)
	 for(int j = 0; j < cols; j++)
            elem[i*rows+j] = 0;
    }

    ~Float2D()
    {
       delete elem;
    }

    inline float* operator[](int i) { 
       return (elem + (cols * i)); 
    }

    inline float const* operator[](int i) const {
       return (elem + (cols * i)); 
    }
    
    inline float* elemVector() {
       return elem;
    }


  private:
    const int rows;
    const int cols;
    float* elem; 
};

//-------- Methods for Visualistion of Results --------

/**
 * generate output filenames for the single-SWE_Block version
 * (for serial and OpenMP-parallelised versions that use only a 
 *  single SWE_Block - one output file is generated per checkpoint)
 */
inline std::string generateFileName(std::string baseName, int timeStep) {

	std::ostringstream FileName;
	FileName << baseName <<timeStep<<".vtk";
	return FileName.str();
};

/**
 * generate output filename for the multiple-SWE_Block version
 * (for serial and parallel (OpenMP and MPI) versions that use 
 *  multiple SWE_Blocks - for each block, one output file is 
 *  generated per checkpoint)
 */
inline std::string generateFileName(std::string baseName, int timeStep, int block_X, int block_Y) {

	std::ostringstream FileName;
	FileName << baseName <<"_"<< block_X<<"_"<<block_Y<<"_"<<timeStep<<".vts";
	return FileName.str();
};

/**
 * generate output filename for the ParaView-Container-File
 * (to visualize multiple SWE_Blocks per checkpoint)
 */
inline std::string generateContainerFileName(std::string baseName, int timeStep) {

	std::ostringstream FileName;
	FileName << baseName<<"_"<<timeStep<<".pvts";
	return FileName.str();
};


#endif

