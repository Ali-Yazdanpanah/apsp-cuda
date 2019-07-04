
/*
 * Max matrix size is N = 2^12 = 4096. 
 * Total number of threads that will be executed is 4096^2.
 * One for each cell.
 */



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
 
// Solves the all-pairs shortest path problem using Floyd Warshall algorithm
void floyd_warshall_parallel(int** graph, int** dist, int N) {

	// allocate memory on the device
	int* device_dist;
	gpuErrchk( cudaMalloc( (void**)&device_dist, N*N * sizeof (int) ) );
	// initialize dist matrix on device
	for (int i = 0; i < N; i++)
		gpuErrchk( cudaMemcpy(device_dist +i*N, graph[i], N * sizeof (int),
							  cudaMemcpyHostToDevice) );
	// For each element of the vertex set
	for (int k = 0; k < N; k++) {
		// launch kernel
		floyd_warshall_parallel_kernel<<<blocks,threads>>>(device_dist,N,k);
		gpuKerchk();
    	}
	// return results to dist matrix on host
	for (int i = 0; i < N; i++)
		 gpuErrchk( cudaMemcpy(dist[i], device_dist +i*N, N * sizeof (int),
							  cudaMemcpyDeviceToHost) );
}
