/* Solve del^2 u = -8*pi^2*sin(2*pi*x)*sin(2*pi*y)
With Dirichlet BCs = 1 on 0<x<1, 0<y<1.
Analytical solution: u = sin(2*pi*x)*sin(2*pi*y) + 1

*/

#include <cusp/gallery/poisson.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include <cusp/krylov/cg.h>
#include <cusp/krylov/bicg.h>
#include <cusp/krylov/gmres.h>

#include <cusp/precond/diagonal.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>

#include <cusp/array1d.h>#
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_utils.h"
//#include <cusp/precond/smoothed_aggregation.h> // FEHLER IM TEMPLATE

double get_time() 
{  struct timeval tim;
  cudaThreadSynchronize();
  gettimeofday(&tim, NULL);
  return (double) tim.tv_sec+(tim.tv_usec/1000000.0);
}

int main()
{
	/*
	*		Wozu brauchen wir die?
	*/ 
	typedef int IndexType;
	typedef float ValueType;
	typedef cusp::device_memory MemorySpace;

	int N = 100; // Nodes per side
	float xmin = 0.0f;
	float xmax = 1.0f;
	float ymin = 0.0f;
	float h = (xmax - xmin)/(float)(N-1);
	
	// Generate mesh (if plotting and for RHS)
	cusp::array1d<float, MemorySpace> x(N*N, 0);
	cusp::array1d<float, MemorySpace> y(N*N, 0);
	for (int j=0; j<N; j++)
	{	for (int i=0; i<N; i++)
		{	x[j*N+i] = xmin + i*h;
			y[j*N+i] = ymin + j*h;
		}
	}

	// Generate matrix empty COO structure
	cusp::coo_matrix<int, float, MemorySpace> A;
	cusp::coo_matrix<int, float, MemorySpace> M;
	//cusp::coo_matrix<int, float, cusp::device_memory> A_d;

	// create 2D Poisson problem
	cusp::gallery::poisson5pt(A, N-2, N-2);

	// Generate RHS, solution vector, and analytical solution	
	cusp::array1d<float, MemorySpace> b(A.num_rows, 1.0f);
	cusp::array1d<float, MemorySpace> u(A.num_rows, 0);
	cusp::array1d<float, MemorySpace> u_an(A.num_rows, 0);
	
	//cusp::array1d<float, cusp::device_memory> b_d(A.num_rows, 1.0f);
	//cusp::array1d<float, cusp::device_memory> u_d(A.num_rows, 0);
	//cusp::array1d<float, cusp::device_memory> u_an(A.num_rows, 0);
	
	for (int j=1; j<N-1; j++)
	{	for (int i=1; i<N-1; i++)
		{	
			b[(j-1)*(N-2)+(i-1)] = 8*M_PI*M_PI*sin(2*M_PI*x[j*N+i])*sin(2*M_PI*y[j*N+i])*h*h;
			u_an[(j-1)*(N-2)+(i-1)] = sin(2*M_PI*x[j*N+i])*sin(2*M_PI*y[j*N+i]) + 1.0f;
			if ((j==1) || (j==N-2))
			{	b[(j-1)*(N-2)+(i-1)] += 1.0f;
			} 
			if ((i==1) || (i==N-2))
			{	b[(j-1)*(N-2)+(i-1)] += 1.0f;
			} 
		}
	}

	float tol = 1e-5;
	int max_iter = 1000;
	std::cout<<"vor dem Monitor"<<std::endl;
	// Setup monitor
	cusp::default_monitor<float> monitor(b, max_iter, tol);
	// Setup preconditioner (identity preconditioner is equivalent to nothing!)

	Timer timer;
	
	double duration = 0.0;
	{
		std::cout << "No Preconditioner + CG" << std::endl;
		initTimer(&timer);
		cusp::krylov::cg(A, u, b, monitor);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}
	
	{
		std::cout << "No Preconditioner + gmres" << std::endl;
		initTimer(&timer);
		cusp::krylov::cg(A, u, b, monitor);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}
	
	{
		std::cout << "No Preconditioner + bicg" << std::endl;
		initTimer(&timer);
		cusp::krylov::cg(A, u, b, monitor);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}
	
	{
		std::cout << "Diagonal Preconditioner + CG" << std::endl;
		initTimer(&timer);
		cusp::precond::diagonal<float, MemorySpace> M(A); 
		cusp::krylov::cg(A, u, b, monitor, M);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}
	
	{
		std::cout << "Diagonal Preconditioner + gmres" << std::endl;
		initTimer(&timer);
		cusp::precond::diagonal<float, MemorySpace> M(A); 
		cusp::krylov::cg(A, u, b, monitor, M);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}
	
	{
		std::cout << "Diagonal Preconditioner + bicg" << std::endl;
		initTimer(&timer);
		cusp::precond::diagonal<float, MemorySpace> M(A); 
		cusp::krylov::cg(A, u, b, monitor, M);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}

	{
		std::cout << "smoothed_aggregation + CG" << std::endl;
		initTimer(&timer);
		cusp::precond::aggregation::smoothed_aggregation<int, float, MemorySpace> M(A); 
		cusp::krylov::cg(A, u, b, monitor, M);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}
	
	{
		std::cout << "smoothed_aggregation + gmres" << std::endl;
		initTimer(&timer);
		cusp::precond::aggregation::smoothed_aggregation<int, float, MemorySpace> M(A); 
		cusp::krylov::cg(A, u, b, monitor, M);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}
	
	{
		std::cout << "smoothed_aggregation + bicg" << std::endl;
		initTimer(&timer);
		cusp::precond::aggregation::smoothed_aggregation<int, float, MemorySpace> M(A); 
		cusp::krylov::cg(A, u, b, monitor, M);
		duration = getAndResetTimer(&timer);
		float L2_error, L2_1 = 0.0f, L2_2 = 0.0f;
		for (int j=1; j<N-1; j++)
		{		for (int i=1; i<N-1; i++)
			{	L2_1 += (u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)])*(u[(j-1)*(N-2)+(i-1)] - u_an[(j-1)*(N-2)+(i-1)]);
				L2_2 += u_an[(j-1)*(N-2)+(i-1)]*u_an[(j-1)*(N-2)+(i-1)];	
			}
		}
		L2_error = sqrt(L2_1/L2_2);
		std::cout<<L2_error <<" Duration: "<<duration<<std::endl;
	}	
	
	
	
//	cusp::precond::diagonal<float, cusp::host_memory> M(A);
	//cusp::array1d<ValueType, MemorySpace> u_an(u);
//	cusp::krylov::cg(A, u, b, monitor);
//	cusp::krylov::cg(A, u, b, monitor, M);
	//cusp::krylov::gmres(A, u, b, monitor, M);
	
/*	for(method=0; method<3; method++)
	{
		for(precond=0; precond<3; precond++)
		{
			
			switch(precond)
			{
				case 0: 
					strcat(printOut,"no precond /t|/t"); 
					break;
				case 1:
					strcat(printOut,"diagonal precond /t|/t"); 
					cusp::precond::diagonal<float, cusp::host_memory> M(A); 
					break;
				case 2:
				
					strcat(printOut,"smooth_agg precond /t|/t"); 
					cusp::precond::aggregation::smoothed_aggregation<int, float, cusp::host_memory> M(A); 
					break;
			}
			
			/*switch(method)
			{
				case 0:
					strcat(printOut,"GMRES-Method"); 
					cusp::krylov::gmres(A, u, b, monitor, M); 
				break;
				case 1:
					strcat(printOut,"CG-Method");
					cusp::krylov::cg(A, u, b, monitor, M);
				break;
				case 2:
					strcat(printOut,"BiCG-Method");
					cusp::krylov::bicg(A, u, b, monitor, M);
				break;
			}
			
			
		}
	}
*/	
	// Solve 	
}
