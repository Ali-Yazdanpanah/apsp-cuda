#ifndef DIJKSTRA_GPU_H
#define DIJKSTRA_GPU_H

#include "graph_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run block-per-source Dijkstra APSP on GPU. Expects dense n×n weights (row-major).
 * Writes n*n distances to dist_out (dist_out[source*n + v] = shortest path source->v). */
void run_dijkstra_gpu(const MatrixGraph *graph, int *dist_out, int block_dim);

#ifdef __CUDACC__
__global__ void dijkstra_sssp_block(
    const int *__restrict__ weights, int *__restrict__ dist, int n, int inf, int base_source);
__global__ void dijkstra_sssp_block_large(
    const int *__restrict__ weights, int *__restrict__ dist,
    unsigned char *__restrict__ visited, int n, int inf, int base_source);
#endif


#ifdef __cplusplus
}
#endif

#endif /* DIJKSTRA_GPU_H */
