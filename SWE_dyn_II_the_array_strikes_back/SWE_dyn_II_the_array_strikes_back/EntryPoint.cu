#include "SWE_handler.h"
#include <iostream>
using namespace std;

//-------- Methods to define initial conditions --------

/**
* function to initialise unknown h (water level):
* scenario: splashing pool (usewith wall boundaries
*/
float splash_height(float x, float y) {
	return 100 + (1.0 - (x + y));
};

int main()
{
	//times:
	StopWatchInterface* timer;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	
	SWE_handler swe(256, 256, 0.005f, 0.005f, 9.81);

	float endSimulation = 0.2f;
	swe.setBathymetry(1.0f);
	swe.setBoundaryType(WALL, WALL, WALL, WALL); swe.setBoundaryLayer();
	swe.setTimestep(0.005f / sqrtf(9.81 * 11));

	int numCheckPoints = 9;

	// compute the checkpoints in time
	float* checkPoint = new float[numCheckPoints + 1];
	for (int cp = 0; cp <= numCheckPoints; cp++) {
		checkPoint[cp] = cp*(endSimulation / numCheckPoints);
	};

	float t = 0.0f;
	for (int i = 1; i <= numCheckPoints; i++)
	{
		//simulation
		sdkStartTimer(&timer);
		t = swe.simulate(t, checkPoint[i]);
		sdkStopTimer(&timer);
	}

	cout << "Time for simulation: " << sdkGetTimerValue(&timer) << endl;
	cout << "Average time for checkpoint: " << sdkGetAverageTimerValue(&timer) << endl;

	sdkDeleteTimer(&timer);
	checkCudaErrors(cudaDeviceReset());
}