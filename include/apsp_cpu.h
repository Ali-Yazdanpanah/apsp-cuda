#ifndef APSP_CPU_H
#define APSP_CPU_H

#include "graph_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Run Floyd-Warshall on the given graph. Writes n*n distances to dist (row-major). */
void floyd_warshall_cpu(const MatrixGraph *graph, int *dist);

/* Run Dijkstra APSP (all-pairs shortest paths). Writes n*n distances to out (row-major). */
void dijkstra_apsp_cpu(const MatrixGraph *graph, int *out);

#ifdef __cplusplus
}
#endif

#endif /* APSP_CPU_H */
