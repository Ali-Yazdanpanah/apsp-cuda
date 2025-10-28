#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <stddef.h>
#include <stdint.h>

#define GRAPH_INFINITY_DEFAULT 999999

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t order;
    int *weights;
    int infinity;
    int max_weight;
} MatrixGraph;

typedef struct {
    size_t order;
    double density;
    int max_weight;
    int infinity;
    uint64_t seed;
    int undirected;
} GraphGenerationConfig;

MatrixGraph generate_random_matrix_graph(const GraphGenerationConfig *cfg);
void free_matrix_graph(MatrixGraph *graph);
void write_graph_to_file(const MatrixGraph *graph, const char *path);
void read_graph_from_file(MatrixGraph *graph, const char *path);

static inline size_t matrix_index(const MatrixGraph *graph, size_t row, size_t col) {
    return row * graph->order + col;
}

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_UTILS_H */
