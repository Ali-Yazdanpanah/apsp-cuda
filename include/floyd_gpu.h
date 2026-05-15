#ifndef FLOYD_GPU_H
#define FLOYD_GPU_H

#include "graph_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run blocked Floyd-Warshall on GPU. Writes n*n distances to dist_out.
 * block_dim must be <= 32. */
void run_floyd_gpu(const MatrixGraph *graph, int *dist_out, int block_dim);

#ifdef __CUDACC__
__global__ void floyd_phase1_pivot(int *__restrict__ dist, int n, int k_block, int inf, int B);
__global__ void floyd_phase2_row_col(int *__restrict__ dist, int n, int k_block, int inf, int B);
__global__ void floyd_phase3_remaining(int *__restrict__ dist, int n, int k_block, int inf, int B);
#endif


#ifdef __cplusplus
}
#endif

#endif /* FLOYD_GPU_H */
