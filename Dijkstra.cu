
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INFTY 99999

int **A_Matrix, **D_Matrix;
int **stpSet;

// A utility function to find the vertex with minimum distance value, from 
// the set of vertices not yet included in shortest path tree 
__device__ int minDistance(int* dist, int* stpSet, int n) 
{ 
   // Initialize min value 
    int min = INFTY, min_index; 
    for(int j = 0; j < n; j++){
        if (stpSet[j] == 0 && dist[j] <= min) 
        min = dist[j], min_index = j;
    } 
    return min_index; 
} 


__device__ void makeAdjacency(int n){   //Set initial values to node distances
    int N=n;
    int i,j;
    A_Matrix = (int **)malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        A_Matrix[i] = (int *)malloc(N * sizeof(int));
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






__global__ void dijkstra_all(int** graph, int** dist, int** stpSet, int n){
    int tid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if(tid < n)
        dijkstra(graph, dist[tid], stpSet[tid], n, tid);
}

// Function that implements Dijkstra's single source shortest path algorithm 
// for a graph represented using adjacency matrix representation 
__device__ void dijkstra(int** graph,int* dist,int* stpSet, int n, int src) 
{      // The output array.  dist[i] will hold the shortest 
                      // distance from src to i 
   
    for (int i = 0; i < n; i++) 
        dist[i] = INFTY, stpSet[i] = 0;  
    dist[src] = 0; 
    for (int count = 0; count < n-1; count++) 
    { 
        // Pick the minimum distance vertex from the set of vertices not 
        // yet processed. u is always equal to src in the first iteration. 
        int u = minDistance(dist, stpSet , n); 
        // Mark the picked vertex as processed 
        stpSet[u] = 1; 
        // Update dist value of the adjacent vertices of the picked vertex. 
        for (int v = 0; v < n; v++) 
                 
        // Update dist[v] only if is not in stpSet, there is an edge from  
        // u to v, and total weight of path from src to  v through u is  
        // smaller than current value of dist[v] 
            if (!stpSet[v] && graph[u][v] && dist[u] != INFTY  
                                    && dist[u]+graph[u][v] < dist[v]) 
                dist[v] = dist[u] + graph[u][v]; 
            } 
} 

/* Serial

// driver program to test above function 
int main(int argc, char **argv) 
{ 
    int N;     
	N = atoi(argv[1]);  //Read the console inputs
    int i, j, k;
	D_Matrix =(int **) malloc(N * sizeof(int *));
    stpSet = (int **) malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        D_Matrix[i] = (int *)malloc(N * sizeof(int));
        stpSet[i] = (int *)malloc(N * sizeof(int));
    }
    makeAdjacency(N);
    clock_t start = clock();  //First time measurement
    // algorithm -->
    dijkstra_all(A_Matrix,D_Matrix,stpSet,N);
    // <--;	
	clock_t end = clock();   //Final time calculation and convert it into seconds
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time = %f sec\n", seconds);
    return 0; 
} 


*/




int main(int argc, char **argv){
    int N;     
	N = atoi(argv[1]);  //Read the console inputs
    int i;
    int *temp;
    temp = (int)malloc(N * sizeof(int));
    for(i = 0; i < N; i++)
    {
        temp[i] = 0; 
    }
	D_Matrix = (int **)malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        D_Matrix[i] = (int *)malloc(N * sizeof(int));
    }
    makeAdjacency(N);
	int gridx = pow(2, N - 4), gridy = pow(2, N - 4);  //Dimensions of grid
	int blockx = pow(2, 4), blocky = pow(2, 4);
	dim3 dimGrid(gridx, gridy);
	dim3 dimBlock(blockx, blocky);
	
	// allocate memory on the device
	int* device_dist;
    cudaMalloc( (void**)&device_dist, N*N * sizeof (int) );
    cudaMalloc( (void**)&device_graph, N*N * sizeof (int));
    cudaMalloc( (void**)&device_stpSet, N*N * sizeof (int));
    
    cudaMemcpy(device_stpSet, temp, N * sizeof(int), cudaMemcpyHostToDevice);
    // initialize dist matrix on device
    for (int i = 0; i < N; i++){
		cudaMemcpy(device_dist +i*N, A_Matrix[i], N * sizeof (int),
                              cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < N; i++){
		cudaMemcpy(device_graph +i*N, A_Matrix[i], N * sizeof (int),
                              cudaMemcpyHostToDevice);
    }
    dijkstra_all(int** graph, int** dist, int** stpSet, int n)
	clock_t start = clock();
	dijkstra_all<<<dimGrid,dimBlock>>>(device_graph,device_dist,device_stpSet,N);
	clock_t end = clock();   //Final time calculation and convert it into seconds
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time on gpu = %f sec\n", seconds);

	// return results to dist matrix on host
	for (int i = 0; i < N; i++)
		 cudaMemcpy(D_Matrix[i], device_dist +i*N, N * sizeof (int),
							  cudaMemcpyDeviceToHost);	
}



