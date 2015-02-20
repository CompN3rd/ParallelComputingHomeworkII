#ifndef WRITEVTK
#define WRITEVTK
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// VTK writer
void writeVTK(char* caseName, int t_iter, float4 *particles, int n) {

    int i;
    FILE *file; 
    char name[100], end[100], s_t_iter[100];

    sprintf(name,"%s_",caseName);

    sprintf(s_t_iter,"%d", t_iter);
    
    strcpy(end,".vtk");

    if (t_iter<10) {
        strcat(name,"0000");
        strcat(name,s_t_iter);
        strcat(name,end);
    } else if (t_iter<100) {
        strcat(name,"000");
        strcat(name,s_t_iter);
        strcat(name,end);
    } else if (t_iter<1000) {
        strcat(name,"00");
        strcat(name,s_t_iter);
        strcat(name,end);
    } else if (t_iter<10000) {
        strcat(name,"0");
        strcat(name,s_t_iter);
        strcat(name,end);
    } else {
        strcat(name,s_t_iter);
        strcat(name,end);
    }

    //printf("Writing file %s...",name);
    
    file = fopen(name,"w");
    fprintf(file,"# vtk DataFile Version 2.0\n");
    fprintf(file,"NBodies \n");
    fprintf(file,"ASCII\n");
    fprintf(file,"DATASET UNSTRUCTURED_GRID\n");
	fprintf(file, "POINTS %d float\n", n);
	for (i = 0; i < n; i++)
	{
		float4 part = particles[i];
		
		fprintf(file, "%g %g %g", part.x, part.y, part.z);
		if (i < n - 1)
			fprintf(file, "\n");
	}
    
    fclose(file);

    //printf("done.\n");
}
#endif
