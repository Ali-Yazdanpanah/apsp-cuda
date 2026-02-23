#ifndef FLOYD_GPU_H
#define FLOYD_GPU_H

#include "graph_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run blocked Floyd-Warshall on GPU. Writes n*n distances to dist_out.
 * block_dim must be <= 32. */
void run_floyd_gpu(const MatrixGraph *graph, int *dist_out, int block_dim);

#ifdef __cplusplus
}
#endif

#endif /* FLOYD_GPU_H */
