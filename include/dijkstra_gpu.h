#ifndef DIJKSTRA_GPU_H
#define DIJKSTRA_GPU_H

#include "graph_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run block-per-source Dijkstra APSP on GPU. Expects dense n×n weights (row-major).
 * Writes n*n distances to dist_out (dist_out[source*n + v] = shortest path source->v). */
void run_dijkstra_gpu(const MatrixGraph *graph, int *dist_out, int block_dim);

#ifdef __cplusplus
}
#endif

#endif /* DIJKSTRA_GPU_H */
