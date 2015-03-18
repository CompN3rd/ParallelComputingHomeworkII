#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define __DRIVER_TYPES_H__
#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_math.h>
#include <iostream>
#include <stdio.h>
#include <fstream>

template <typename MemoryType>
class ArrayHelper
{
private:
	int nx;
	int ny;

	//array for values
	MemoryType* values;

public:
	ArrayHelper(int x, int y)
	{
		this->nx = x;
		this->ny = y;

		values = (MemoryType*)malloc(nx * ny * sizeof(MemoryType));
		memset(values, 0, nx * ny * sizeof(MemoryType));
	}

	~ArrayHelper()
	{
		free(values);
	}

	inline MemoryType* getValues()
	{
		return values;
	}

	inline int getWidth()
	{
		return this->nx;
	}

	inline int getHeight()
	{
		return this->ny;
	}
};