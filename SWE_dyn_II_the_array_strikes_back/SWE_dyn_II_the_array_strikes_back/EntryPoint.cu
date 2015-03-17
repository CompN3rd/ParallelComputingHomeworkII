#include "SWE_handler.h"
using namespace std;

//-------- Methods to define initial conditions --------

/**
* function to initialise unknown h (water level):
* scenario: splashing pool (usewith wall boundaries
*/
float splash_height(float x, float y) {
	return 100.0f + (1.0f - (x + y));
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
	
	SWE_handler swe(256, 256, 0.005f, 0.005f, 9.81f);

	float endSimulation = 0.1f;
	swe.setBathymetry(1.0f);
	swe.setBoundaryType(WALL, WALL, WALL, WALL); swe.setBoundaryLayer();
	//swe.setTimestep(0.005f / sqrtf(9.81f * 11.0f));
	swe.setTimestep(0.2f);

	int numCheckPoints = 1;

	// compute the checkpoints in time
	float* checkPoint = new float[numCheckPoints + 1];
	for (int cp = 0; cp <= numCheckPoints; cp++) {
		checkPoint[cp] = cp*(endSimulation / numCheckPoints);
	};

	//output the initial solution
	cout << "Writing solution at time: " << checkPoint[0] << endl;
	swe.writeVTKFile(generateFileName(basename, 0));

	float t = 0.0f;
	for (int i = 1; i <= numCheckPoints; i++)
	{
		//simulation
		sdkStartTimer(&timer);
		t = swe.simulate(t, checkPoint[i]);
		checkCudaErrors(cudaDeviceSynchronize());
		sdkStopTimer(&timer);

		//write result
		cout << "Writing solution at time: " << checkPoint[i] << endl;
		swe.writeVTKFile(generateFileName(basename, i));
	}

	cout << "Time for simulation: " << sdkGetTimerValue(&timer) / 1000.0f << " sec" << endl;
	cout << "Average time for checkpoint: " << sdkGetAverageTimerValue(&timer) / 1000.0f << " sec" << endl;

	sdkDeleteTimer(&timer);
	checkCudaErrors(cudaDeviceReset());
}