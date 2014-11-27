#ifndef WRITEVTK
#define WRITEVTK

// VTK writer
void writeVTK(char* caseName, int t_iter, int nx, int ny, float *u) {

    int i, j;
    FILE *file; 
    char name[100], end[100], s_t_iter[100];

    sprintf(name,"FD_2D_%s_",caseName);

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

    printf("Writing file %s...",name);
    
    file = fopen(name,"w");
    fprintf(file,"# vtk DataFile Version 2.0\n");
    fprintf(file,"heat\n");
    fprintf(file,"ASCII\n");
    fprintf(file,"DATASET STRUCTURED_POINTS\n");
    fprintf(file,"DIMENSIONS %d %d 1\n", nx, ny);
    fprintf(file,"SPACING 1 1 1\n");
    fprintf(file,"ORIGIN %f %f 0\n", 0.0, 0.0);
    fprintf(file,"POINT_DATA %d\n", nx*nx);
    fprintf(file,"SCALARS temperature float 1\n");
    fprintf(file,"LOOKUP_TABLE default\n");
    for (j=0; j<ny; j++) {
        for (i=0; i<nx; i++) {
            fprintf(file,"%g ", u[i+j*nx]);
        }
        fprintf(file,"\n");
    }
    
    fclose(file);

    printf("done.\n");
}
#endif
