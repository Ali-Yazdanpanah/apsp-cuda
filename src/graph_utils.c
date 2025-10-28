#include "graph_utils.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t lcg_next(uint64_t *state) {
    *state = (*state * 2862933555777941757ULL + 3037000493ULL) & 0xFFFFFFFFFFFFFFFFULL;
    return *state;
}

static double lcg_uniform(uint64_t *state) {
    return (lcg_next(state) >> 11) * (1.0 / 9007199254740992.0); /* 53-bit resolution */
}

static int clamp_positive(int value, int fallback) {
    return value > 0 ? value : fallback;
}

MatrixGraph generate_random_matrix_graph(const GraphGenerationConfig *cfg) {
    MatrixGraph graph = {0};
    if (!cfg || cfg->order == 0) {
        return graph;
    }

    graph.order = cfg->order;
    graph.infinity = clamp_positive(cfg->infinity, GRAPH_INFINITY_DEFAULT);
    graph.max_weight = clamp_positive(cfg->max_weight, 100);
    graph.weights = (int *)calloc(graph.order * graph.order, sizeof(int));
    if (!graph.weights) {
        graph.order = 0;
        return graph;
    }

    const double density = fmin(fmax(cfg->density, 0.0), 1.0);
    uint64_t state = cfg->seed ? cfg->seed : 0xC0FFEEULL;

    for (size_t row = 0; row < graph.order; ++row) {
        for (size_t col = 0; col < graph.order; ++col) {
            const size_t idx = matrix_index(&graph, row, col);
            if (row == col) {
                graph.weights[idx] = 0;
            } else {
                graph.weights[idx] = graph.infinity;
            }
        }
    }

    if (density <= 0.0) {
        return graph;
    }

    if (cfg->undirected) {
        for (size_t row = 0; row < graph.order; ++row) {
            for (size_t col = row + 1; col < graph.order; ++col) {
                double sample = lcg_uniform(&state);
                if (sample <= density) {
                    int weight = 1 + (int)(lcg_next(&state) % (unsigned)graph.max_weight);
                    graph.weights[matrix_index(&graph, row, col)] = weight;
                    graph.weights[matrix_index(&graph, col, row)] = weight;
                }
            }
        }
    } else {
        for (size_t row = 0; row < graph.order; ++row) {
            for (size_t col = 0; col < graph.order; ++col) {
                if (row == col) {
                    continue;
                }
                double sample = lcg_uniform(&state);
                if (sample <= density) {
                    int weight = 1 + (int)(lcg_next(&state) % (unsigned)graph.max_weight);
                    graph.weights[matrix_index(&graph, row, col)] = weight;
                }
            }
        }
    }

    return graph;
}

void free_matrix_graph(MatrixGraph *graph) {
    if (!graph) {
        return;
    }
    free(graph->weights);
    graph->weights = NULL;
    graph->order = 0;
    graph->infinity = GRAPH_INFINITY_DEFAULT;
    graph->max_weight = 0;
}

void write_graph_to_file(const MatrixGraph *graph, const char *path) {
    if (!graph || !graph->weights || !path) {
        return;
    }
    FILE *file = fopen(path, "w");
    if (!file) {
        fprintf(stderr, "Failed to open %s for writing: %s\n", path, strerror(errno));
        return;
    }
    fprintf(file, "%zu %d %d\n", graph->order, graph->infinity, graph->max_weight);
    for (size_t row = 0; row < graph->order; ++row) {
        for (size_t col = 0; col < graph->order; ++col) {
            fprintf(file, "%d", graph->weights[matrix_index(graph, row, col)]);
            if (col + 1 < graph->order) {
                fputc(' ', file);
            }
        }
        fputc('\n', file);
    }
    fclose(file);
}

void read_graph_from_file(MatrixGraph *graph, const char *path) {
    if (!graph || !path) {
        return;
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open %s for reading: %s\n", path, strerror(errno));
        return;
    }

    size_t order = 0;
    int infinity = 0;
    int max_weight = 0;
    if (fscanf(file, "%zu %d %d", &order, &infinity, &max_weight) != 3) {
        fprintf(stderr, "Invalid graph header in %s\n", path);
        fclose(file);
        return;
    }

    MatrixGraph tmp = {
        .order = order,
        .infinity = clamp_positive(infinity, GRAPH_INFINITY_DEFAULT),
        .max_weight = clamp_positive(max_weight, 100),
        .weights = (int *)malloc(order * order * sizeof(int)),
    };
    if (!tmp.weights) {
        fprintf(stderr, "Allocation failed while reading graph from %s\n", path);
        fclose(file);
        return;
    }

    for (size_t row = 0; row < order; ++row) {
        for (size_t col = 0; col < order; ++col) {
            if (fscanf(file, "%d", &tmp.weights[matrix_index(&tmp, row, col)]) != 1) {
                fprintf(stderr, "Invalid graph data in %s\n", path);
                free(tmp.weights);
                fclose(file);
                return;
            }
        }
    }

    fclose(file);
    free_matrix_graph(graph);
    *graph = tmp;
}
