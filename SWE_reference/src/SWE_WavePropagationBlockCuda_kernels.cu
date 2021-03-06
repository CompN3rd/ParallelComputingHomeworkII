/**
 * @file
 * This file is part of SWE.
 *
 * @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
 * @author Sebastian Rettenberger (rettenbs AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
 *
 * @section LICENSE
 *
 * SWE is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SWE is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SWE.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * @section DESCRIPTION
 *
 * CUDA Kernels for a SWE_Block, which uses solvers in the wave propagation formulation.
 */

#include "SWE_BlockCUDA.hh"
#include "SWE_WavePropagationBlockCuda_kernels.hh"

#include <cmath>
#include <cstdio>

#include "solvers/FWaveCuda.h"

/**
 * The compute net-updates kernel calls the solver for a defined CUDA-Block and does a reduction over the computed wave speeds within this block.
 *
 * Remark: In overall we have nx+1 / ny+1 edges.
 *         Therefore the edges "simulation domain"/"top ghost layer" and "simulation domain"/"right ghost layer"
 *         will not be computed in a typical call of the function:
 *           computeNetUpdatesKernel<<<dimGrid,dimBlock>>>( hd, hud, hvd, bd,
 *                                                          hNetUpdatesLeftD,  hNetUpdatesRightD,
 *                                                          huNetUpdatesLeftD, huNetUpdatesRightD,
 *                                                          hNetUpdatesBelowD, hNetUpdatesAboveD,
 *                                                          hvNetUpdatesBelowD, hvNetUpdatesAboveD,
 *                                                          l_maximumWaveSpeedsD,
 *                                                          i_nx, i_ny
 *                                                        );
 *         To reduce the effect of branch-mispredictions the kernel provides optional offsets, which can be used to compute the missing edges.
 *
 * {@link SWE_WavePropagationBlockCuda::computeNumericalFluxes()} explains the coalesced memory access.
 *
 * @param i_h water heights (CUDA-array).
 * @param i_hu momentums in x-direction (CUDA-array).
 * @param i_hv momentums in y-direction (CUDA-array).
 * @param i_b bathymetry values (CUDA-array).
 * @param o_hNetUpdatesLeftD left going net-updates for the water height (CUDA-array).
 * @param o_hNetUpdatesRightD right going net-updates for the water height (CUDA-array).
 * @param o_huNetUpdatesLeftD left going net-updates for the momentum in x-direction (CUDA-array).
 * @param o_huNetUpdatesRightD right going net-updates for the momentum in x-direction (CUDA-array).
 * @param o_hNetUpdatesBelowD downwards going net-updates for the water height (CUDA-array).
 * @param o_hNetUpdatesAboveD upwards going net-updates for the water height (CUDA-array).
 * @param o_hvNetUpdatesBelowD downwards going net-updates for the momentum in y-direction (CUDA-array).
 * @param o_hvNetUpdatesAboveD upwards going net-updates for the momentum in y-direction (CUDA-array).
 * @param o_maximumWaveSpeeds maximum wave speed which occurred within the CUDA-block is written here (CUDA-array).
 * @param i_nx number of cells within the simulation domain in x-direction (excludes ghost layers).
 * @param i_ny number of cells within the simulation domain in y-direction (excludes ghost layers).
 * @param i_offsetX cell/edge offset in x-direction.
 * @param i_offsetY cell/edge offset in y-direction.
 */
__global__
void computeNetUpdatesKernel(
    const float* i_h, const float* i_hu, const float* i_hv, const float* i_b,
    float* o_hNetUpdatesLeftD,   float* o_hNetUpdatesRightD,
    float* o_huNetUpdatesLeftD,  float* o_huNetUpdatesRightD,
    float* o_hNetUpdatesBelowD,  float* o_hNetUpdatesAboveD,
    float* o_hvNetUpdatesBelowD, float* o_hvNetUpdatesAboveD,
    float* o_maximumWaveSpeeds,
    const int i_nX, const int i_nY,
    const int i_offsetX, const int i_offsetY,
    const int i_blockOffSetX, const int i_blockOffSetY
) {
  /*
   * Initialization
   */

  //! index (l_cellIndexI,l_cellIndexJ) of the cell lying on the right side of the edge/above the edge where the thread works on.
  int l_cellIndexI, l_cellIndexJ;
  
    /**
	* Position of the maximum wave speed in the global device array.
	*
	* In the 'main' part (e.g. gridDim = nx/TILE_SIZEm ny/TILE_SIZE) the position
	* is simply given by the blockId in x- and y-direction with a stride of gridDim.x + 1. The +1 results from the
	* speeds in the 'boundary' case, see below.
	*
	* In the 'boundary' case, where the edges lie between the computational domain and the right/top ghost layer,
	* this is more complicated. In this case block offsets in x- and y-direction are used. The offsets define how
	* many blocks in the resp. direction have to be added to get a valid result.
	* Computational domain - right ghost layer: In this case the dimension of the grid in x-direction is 1.
	* Computational domain - top ghost layer: In this case the dimension of the grid in y-direction is 1.
	*
	* Same Example as in SWE_WavePropagationBlockCuda::computeNumericalFluxes(), assume the CUDA-grid/-blocks has
	* the following layout:
	* <pre>
	* *
	* ** top ghost layer,
	* * block 8 * block 9 * block 10* ******** cell ids
	* ******************************** **
	* * * * * *
	* * block * block * block * b
	* * 4 * 5 * 6 * 7
	* * * * *
	* ********************************
	* * * * *
	* * block * block * block * b
	* * * 0 * 1 * 2 * 3
	* bottom ** * * * *
	* ghost ******** ********************************
	* layer **
	* * * *
	* *** ***
	* * *
	* * *
	* left ghost layer right ghost layer
	* </pre>
	* This results in a 'main' part containing of (3*2) blocks and two 'boundary' parts containing
	* of (1*2) blocks and (3*1) blocks.
	*
	* The maximum wave speed array on the device represents therefore logically a (4 * 3)-1 2D-array (-1: no block on the top right).
	* The 'main' part writes into cells 0, 1, 2, 4, 5 and 6.
	* The 'computational domain - right ghost layer' part writes into 3 and 7 with offset in x-direction = 3
	* The 'computational domain - top ghost layer' part writes into 8, 9, 10 with offset in y-direction = 2
	*
	*/
    int l_maxWaveSpeedDevicePosition = computeOneDPositionKernel( i_blockOffSetX + blockIdx.x,
                                                                  i_blockOffSetY + blockIdx.y,
                                                                  max(i_blockOffSetY+1, gridDim.y+1) );

  // initialize the indices l_cellIndexI and l_cellIndexJ with the given offset
  l_cellIndexI = i_offsetX;
  l_cellIndexJ = i_offsetY;

  //! array which holds the thread local net-updates.
  float l_netUpdates[5];

  //! location of the thread local cells in the global CUDA-arrays.
  int l_leftCellPosition, l_rightCellPosition, l_netUpdatePosition;

  // compute (l_cellIndexI,l_cellIndexJ) for the cell lying right/above of the edge
  l_cellIndexI += blockDim.y * blockIdx.x + threadIdx.y + 1; //+1: start at cell with index (1,0)
  l_cellIndexJ += blockDim.x * blockIdx.y + threadIdx.x + 1; //+1: start at cell with index (0,1)

  l_rightCellPosition = computeOneDPositionKernel(l_cellIndexI, l_cellIndexJ, i_nY+2);

  //allocate shared memory for the state vectors
  __shared__ float s_h[TILE_SIZE + 1][TILE_SIZE + 1];
  __shared__ float s_hu[TILE_SIZE + 1][TILE_SIZE + 1];
  __shared__ float s_hv[TILE_SIZE + 1][TILE_SIZE + 1];
  __shared__ float s_b[TILE_SIZE + 1][TILE_SIZE + 1];

  //Each thread loads a state vector into shared memory
  s_h[threadIdx.y + 1][threadIdx.x + 1] = i_h[l_rightCellPosition];
  s_hu[threadIdx.y + 1][threadIdx.x + 1] = i_hu[l_rightCellPosition];
  s_hv[threadIdx.y + 1][threadIdx.x + 1] = i_hv[l_rightCellPosition];
  s_b[threadIdx.y + 1][threadIdx.x + 1] = i_b[l_rightCellPosition];

  //Threads with x-index 0 have to load the left-most column of the block as well
  if (threadIdx.x == 0) {
	l_leftCellPosition = computeOneDPositionKernel(l_cellIndexI, l_cellIndexJ-1, i_nY+2);

	s_h[threadIdx.y + 1][0] = i_h[l_leftCellPosition];
	s_hu[threadIdx.y + 1][0] = i_hu[l_leftCellPosition];
	s_hv[threadIdx.y + 1][0] = i_hv[l_leftCellPosition];
	s_b[threadIdx.y + 1][0] = i_b[l_leftCellPosition];
  }

  //Threads with y-index 0 have to load the bottom-most row of the block as well
  if (threadIdx.y == 0) {
	l_leftCellPosition = computeOneDPositionKernel(l_cellIndexI-1, l_cellIndexJ, i_nY+2);

	s_h[0][threadIdx.x + 1] = i_h[l_leftCellPosition];
	s_hu[0][threadIdx.x + 1] = i_hu[l_leftCellPosition];
	s_hv[0][threadIdx.x + 1] = i_hv[l_leftCellPosition];
	s_b[0][threadIdx.x + 1] = i_b[l_leftCellPosition];
  }

  //Wait until all threads have loaded their data into shared memory
  __syncthreads();

  /*
   * Computation of horizontal net-updates
   */
  if(i_offsetY == 0) {
    // compute the net-updates
    fWaveComputeNetUpdates( 9.81,
                            s_h[threadIdx.y][threadIdx.x + 1],  s_h[threadIdx.y + 1][threadIdx.x + 1],
                            s_hu[threadIdx.y][threadIdx.x + 1], s_hu[threadIdx.y + 1][threadIdx.x + 1],
                            s_b[threadIdx.y][threadIdx.x + 1],  s_b[threadIdx.y + 1][threadIdx.x + 1],
                            l_netUpdates
                           );

    // compute the location of the net-updates in the global CUDA-arrays.
    l_netUpdatePosition = computeOneDPositionKernel(l_cellIndexI-1, l_cellIndexJ, i_nY+1);

    // store the horizontal net-updates (thread local net-updates -> device net-updates)
    o_hNetUpdatesLeftD[l_netUpdatePosition]  = l_netUpdates[0]; o_hNetUpdatesRightD[l_netUpdatePosition]  = l_netUpdates[1];
    o_huNetUpdatesLeftD[l_netUpdatePosition] = l_netUpdates[2]; o_huNetUpdatesRightD[l_netUpdatePosition] = l_netUpdates[3];

    // update maximum wave speed in the block
    o_maximumWaveSpeeds[l_maxWaveSpeedDevicePosition] = max(o_maximumWaveSpeeds[l_maxWaveSpeedDevicePosition], l_netUpdates[4]);
  }

  // synchronize the threads before the vertical updates (optimization)
  __syncthreads();

  /*
   * Computation of vertical net-updates
   */
  if(i_offsetX == 0) {
    // compute the net-updates
    fWaveComputeNetUpdates( 9.81,
                            s_h[threadIdx.y + 1][threadIdx.x],  s_h[threadIdx.y + 1][threadIdx.x + 1],
                            s_hv[threadIdx.y + 1][threadIdx.x], s_hv[threadIdx.y + 1][threadIdx.x + 1],
                            s_b[threadIdx.y + 1][threadIdx.x],  s_b[threadIdx.y + 1][threadIdx.x + 1],
                            l_netUpdates
                           );

    // compute the location of the net-updates in the global CUDA-arrays.
    l_netUpdatePosition = computeOneDPositionKernel(l_cellIndexI, l_cellIndexJ-1, i_nY+1);

    // store the vertical net-updates (thread local net-updates -> device net-updates)
    o_hNetUpdatesBelowD[l_netUpdatePosition]  = l_netUpdates[0]; o_hNetUpdatesAboveD[l_netUpdatePosition]  = l_netUpdates[1];
    o_hvNetUpdatesBelowD[l_netUpdatePosition] = l_netUpdates[2]; o_hvNetUpdatesAboveD[l_netUpdatePosition] = l_netUpdates[3];

    // update maximum wave speed in the block
    o_maximumWaveSpeeds[l_maxWaveSpeedDevicePosition] = max(o_maximumWaveSpeeds[l_maxWaveSpeedDevicePosition], l_netUpdates[4]);
  }
}

/**
 * The "update unknowns"-kernel updates the unknowns in the cells with precomputed net-updates.
 *
 * {@link SWE_WavePropagationBlockCuda::computeNumericalFluxes()} explains the coalesced memory access.
 *
 * @param i_hNetUpdatesLeftD left going net-updates for the water height (CUDA-array).
 * @param i_hNetUpdatesRightD right going net-updates for the water height (CUDA-array).
 * @param i_huNetUpdatesLeftD left going net-updates for the momentum in x-direction (CUDA-array).
 * @param i_huNetUpdatesRightD right going net-updates for the momentum in x-direction (CUDA-array).
 * @param i_hNetUpdatesBelowD downwards going net-updates for the water height (CUDA-array).
 * @param i_hNetUpdatesAboveD upwards going net-updates for the water height (CUDA-array).
 * @param i_hvNetUpdatesBelowD downwards going net-updates for the momentum in y-direction (CUDA-array).
 * @param i_hvNetUpdatesAboveD upwards going net-updates for the momentum in y-direction (CUDA-array).
 * @param io_h water heights (CUDA-array).
 * @param io_hu momentums in x-direction (CUDA-array).
 * @param io_hv momentums in y-direction (CUDA-array).
 * @param i_updateWidthX update width in x-direction.
 * @param i_updateWidthY update width in y-direction.
 * @param i_nx number of cells within the simulation domain in x-direction (excludes ghost layers).
 * @param i_ny number of cells within the simulation domain in y-direction (excludes ghost layers).
 */
__global__
void updateUnknownsKernel(
    const float* i_hNetUpdatesLeftD,   const float* i_hNetUpdatesRightD,
    const float* i_huNetUpdatesLeftD,  const float* i_huNetUpdatesRightD,
    const float* i_hNetUpdatesBelowD,  const float* i_hNetUpdatesAboveD,
    const float* i_hvNetUpdatesBelowD, const float* i_hvNetUpdatesAboveD,
    float* io_h, float* io_hu, float* io_hv,
    const float i_updateWidthX, const float i_updateWidthY,
    const int i_nX, const int i_nY ) {
  /*
   * Initialization
   */
  //! cell indices (l_cellIndexI,l_cellIndexJ) of the cell which the thread updates.
  int l_cellIndexI, l_cellIndexJ;

  //! location of the thread local cell in the global CUDA-arrays.
  int l_cellPosition;

  // compute the thread local cell indices (start at cell (1,1))
  l_cellIndexI = blockDim.y * blockIdx.x + threadIdx.y + 1;
  l_cellIndexJ = blockDim.x * blockIdx.y + threadIdx.x + 1;

  // compute the global cell position
  l_cellPosition = computeOneDPositionKernel(l_cellIndexI, l_cellIndexJ, i_nY+2);

#ifndef NDEBUG
#if defined(__CUDA_ARCH__) & (__CUDA_ARCH__ < 200)
#warning skipping printf command, which was introduced for compute capability >= 2.0
#else
  if(l_cellPosition > (i_nX+2)*(i_nY+2))
    printf("Warning: cellPosition(%i) > (i_nx+2)*(i_ny+2)\n", l_cellPosition);
#endif
#endif

  //! positions of the net-updates in the global CUDA-arrays.
  int l_netUpdatesLeftPosition, l_netUpdatesRightPosition, l_netUpdatesBelowPosition, l_netUpdatesAbovePosition;

  /**
   *  Compute the positions of the net updates relative to a given cell
   *
   * <pre>
   *                  netUpdateRight(i-1, j)
   *
   *                           |           |
   *                           |           |
   *                           |           |
   *  netUpdateBelow(i,j) -----*************-----
   *                           *           *
   *                           * cell(i,j) *
   *                           *           *
   *                      -----*************------ netUpdateAbove(i, j-1)
   *                           |           |
   *                           |           |
   *                           |           |
   *                                netUpdatesLeft(i,j)
   * </pre>
   */
  l_netUpdatesRightPosition = computeOneDPositionKernel(l_cellIndexI-1, l_cellIndexJ, i_nY+1);
  l_netUpdatesLeftPosition  = computeOneDPositionKernel(l_cellIndexI,   l_cellIndexJ, i_nY+1);

  l_netUpdatesAbovePosition = computeOneDPositionKernel(l_cellIndexI, l_cellIndexJ-1, i_nY+1);
  l_netUpdatesBelowPosition = computeOneDPositionKernel(l_cellIndexI, l_cellIndexJ,   i_nY+1);

  //update the cell values
  io_h[l_cellPosition] -=  i_updateWidthX * ( i_hNetUpdatesRightD[ l_netUpdatesRightPosition ] +  i_hNetUpdatesLeftD[  l_netUpdatesLeftPosition ] )
                         + i_updateWidthY * ( i_hNetUpdatesAboveD[ l_netUpdatesAbovePosition ] +  i_hNetUpdatesBelowD[ l_netUpdatesBelowPosition ] );

  io_hu[l_cellPosition] -= i_updateWidthX * ( i_huNetUpdatesRightD[ l_netUpdatesRightPosition ] + i_huNetUpdatesLeftD[ l_netUpdatesLeftPosition ] );

  io_hv[l_cellPosition] -= i_updateWidthY * ( i_hvNetUpdatesAboveD[ l_netUpdatesAbovePosition ] + i_hvNetUpdatesBelowD[ l_netUpdatesBelowPosition ] );
}

/**
 * Compute the position of 2D coordinates in a 1D array.
 *   array[i][j] -> i * ny + j
 *
 * @param i_i row index.
 * @param i_j column index.
 * @param i_ny #(cells in y-direction).
 * @return 1D index.
 */
__device__
inline int computeOneDPositionKernel(const int i_i, const int i_j, const int i_ny) {
  return i_i*i_ny + i_j;
}
