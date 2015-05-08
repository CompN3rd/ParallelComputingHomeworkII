#include "SWE_handler.h"
using namespace std;

//-------- Methods to define initial conditions --------

/**
* function to initialise unknown h (water level):
* scenario: splashing pool (usewith wall boundaries
*/
float splash_height(float x, float y) {
	//return 100.0f + (1.0f - (x + y));
	return 2.0f - (1.0f / 512.0f) * (x + y);
};

int main(int argc, char** argv)
{
	std::string basename;
	if (argc <= 1) 
	{
		cout << "Please provide filename for output (format: name or /path/name)" << endl;
		cin >> basename;
	}
	else
	{
		basename = std::string(argv[1]);
	}

	//times:
	StopWatchInterface* timer;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	
	SWE_handler* swe = new SWE_handler(256, 256, 0.005f, 0.005f, 9.81f, 4, 4);

	float endSimulation = 0.2f;

	swe->setInitialValues(splash_height, 0.0f, 0.0f);
	swe->setBathymetry(1.0f);
	swe->setBoundaryType(WALL, WALL, WALL, WALL); swe->setBoundaryLayer();
	swe->setTimestep(0.005f / sqrtf(9.81f * 11.0f));
	//swe->setTimestep(0.0005f / sqrtf(9.81f * 11.0f));

	int numCheckPoints = 9;

	// compute the checkpoints in time
	float* checkPoint = new float[numCheckPoints + 1];
	for (int cp = 0; cp <= numCheckPoints; cp++) {
		checkPoint[cp] = cp*(endSimulation / numCheckPoints);
	};

	//output the initial solution
	cout << "Writing solution at time: " << checkPoint[0] << endl;
	swe->synchronizeSolution();
	swe->writeVTKFile(generateFileName(basename, 0));

	float t = 0.0f;
	for (int i = 1; i <= numCheckPoints; i++)
	{
		//simulation
		sdkStartTimer(&timer);
		t = swe->simulate(t, checkPoint[i]);
		swe->synchronizeSolution();
		sdkStopTimer(&timer);

		//write result
		cout << "Writing solution at time: " << checkPoint[i] << endl;
		swe->writeVTKFile(generateFileName(basename, i));
	}

	cout << "Time for simulation: " << sdkGetTimerValue(&timer) / 1000.0f << " sec" << endl;
	cout << "Average time for checkpoint: " << sdkGetAverageTimerValue(&timer) / 1000.0f << " sec" << endl;

	//clean up
	sdkDeleteTimer(&timer);
	delete[] checkPoint;
	delete swe;

	//for profiling
	checkCudaErrors(cudaDeviceReset());
}