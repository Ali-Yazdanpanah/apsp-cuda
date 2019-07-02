
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "floyd_Warshall_serial.h"

#define INFTY 99999

int **A_Matrix, **D_Matrix;




__global__ void floyd_warshall_parallel_kernel(int* dev_dist, int N, int k) {\
	int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int i, j;
	int dist1, dist2, dist3;
	if (tid < N*N) {
		i = tid / N;
		j = tid - i*N;
		dist1 = dev_dist[tid];
		dist2 = dev_dist[i*N +k];
		dist3 = dev_dist[k*N +j];
		if (dist1 > dist2 + dist3)
			dev_dist[tid] = dist2 + dist3;
	}
}


void makeAdjacency(int n){   //Set initial values to node distances
    int N=n;
    int i,j;
    A_Matrix = malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        A_Matrix[i] = malloc(N * sizeof(int));
    }

    srand(0);
    for(i = 0; i < N; i++)
    {
        for(j = i; j < N; j++)
        {
            if(i == j){
                A_Matrix[i][j] = 0;
            }
            else{
                int r = rand() % 10;
                int val = (r == 5)? INFTY: r;
                A_Matrix[i][j] = val;
                A_Matrix[j][i] = val; 
            }

        }
    }
}

int main(int argc, char **argv){
    int N;     
	N = atoi(argv[1]);  //Read the console inputs
    int i, j, k;
	D_Matrix = malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        D_Matrix[i] = malloc(N * sizeof(int));
    }
    makeAdjacency(N);
    clock_t start = clock();  //First time measurement
    // algorithm -->
    floyd_warshall_serial(A_Matrix,D_Matrix,N);
    // <--;	
	clock_t end = clock();   //Final time calculation and convert it into seconds
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time = %f sec\n", seconds);
	int gridx = pow(2, N - 4), gridy = pow(2, N - 4);  //Dimensions of grid
	int blockx = pow(2, 4), blocky = pow(2, 4);
	dim3 dimGrid(gridx, gridy);
	dim3 dimBlock(blockx, blocky);
	
	
	// allocate memory on the device
	int* device_dist;
	gpuErrchk( cudaMalloc( (void**)&device_dist, N*N * sizeof (int) ) );
	// initialize dist matrix on device
	for (int i = 0; i < N; i++)
		gpuErrchk( cudaMemcpy(device_dist +i*N, graph[i], N * sizeof (int),
							  cudaMemcpyHostToDevice) );
	// For each element of the vertex set
	
	clock_t start = clock();
	for (int k = 0; k < N; k++) {
		// launch kernel
		floyd_warshall_parallel_kernel<<<dimGrid,dimBlock>>>(device_dist,N,k);
		gpuKerchk();
    	}
	clock_t end = clock();   //Final time calculation and convert it into seconds
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time on gpu = %f sec\n", seconds);

	// return results to dist matrix on host
	for (int i = 0; i < N; i++)
		 gpuErrchk( cudaMemcpy(dist[i], device_dist +i*N, N * sizeof (int),
							  cudaMemcpyDeviceToHost) );

	
}



