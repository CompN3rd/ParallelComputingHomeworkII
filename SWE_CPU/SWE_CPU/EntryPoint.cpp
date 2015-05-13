#include "SWE.h"

float splash_height(float x, float y)
{
	//return 100 + (1 - (x + y));
	return 2.0f - (1.0f / 512.0f) * (x + y);
}


int main(int argc, char** argv)
{
	SWE swe(256, 256, 0.005f, 0.005f);
	swe.setInitialValues(&splash_height, 0.0f, 0.0f);
	swe.setBathymetry(1.0f);
	swe.setBoundaryType(WALL, WALL, WALL, WALL);
	swe.setTimestep(0.005 / sqrt(9.81 * 11));

	float endSimulation = 0.2f;
	int numCheckPoints = 9;

	float* checkPt = new float[numCheckPoints + 1];
	for (int cp = 0; cp <= numCheckPoints; cp++)
		checkPt[cp] = cp*(endSimulation / numCheckPoints);

	string basename;
	if (argc <= 1)
	{
		cout << "Please provide filename for output (format: name or /path/name)" << endl;
		exit(1);
	}
	else
	{
		basename = string(argv[1]);
	}

	cout << "Write output file: water level at start" << endl;
	swe.writeVTKFile(swe.generateFileName(basename, 0));

	float t = 0.0f;
	for (int i = 1; i <= numCheckPoints; i++)
	{
		t = swe.simulate(t, checkPt[i]);
		cout << "Write output file: water level at time " << t << endl;
		swe.writeVTKFile(swe.generateFileName(basename, i));
	}
}