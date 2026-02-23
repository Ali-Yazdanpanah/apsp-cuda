#define _POSIX_C_SOURCE 200809L

#include "apsp_cpu.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void floyd_warshall_cpu(const MatrixGraph *graph, int *dist) {
    const size_t n = graph->order;
    const int inf = graph->infinity;

    for (size_t i = 0; i < n * n; ++i) {
        dist[i] = graph->weights[i];
    }

    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
            const int dik = dist[matrix_index(graph, i, k)];
            if (dik == inf) {
                continue;
            }
            for (size_t j = 0; j < n; ++j) {
                const int dkj = dist[matrix_index(graph, k, j)];
                if (dkj == inf) {
                    continue;
                }
                const int idx = matrix_index(graph, i, j);
                const long candidate = (long)dik + (long)dkj;
                if (candidate < dist[idx]) {
                    dist[idx] = (int)candidate;
                }
            }
        }
    }
}

void dijkstra_apsp_cpu(const MatrixGraph *graph, int *out) {
    const size_t n = graph->order;
    const int inf = graph->infinity;
    int *dist = (int *)malloc(n * sizeof(int));
    unsigned char *visited = (unsigned char *)malloc(n);

    if (!dist || !visited) {
        free(dist);
        free(visited);
        return;
    }

    for (size_t source = 0; source < n; ++source) {
        for (size_t i = 0; i < n; ++i) {
            dist[i] = inf;
            visited[i] = 0;
        }
        dist[source] = 0;

        for (size_t iter = 0; iter < n; ++iter) {
            size_t u = SIZE_MAX;
            int best_dist = inf;
            for (size_t v = 0; v < n; ++v) {
                if (!visited[v] && dist[v] < best_dist) {
                    best_dist = dist[v];
                    u = v;
                }
            }
            if (u == SIZE_MAX) {
                break;
            }
            visited[u] = 1;

            for (size_t v = 0; v < n; ++v) {
                const int weight = graph->weights[matrix_index(graph, u, v)];
                if (weight >= inf) {
                    continue;
                }
                const long candidate = (long)best_dist + (long)weight;
                if (candidate < dist[v]) {
                    dist[v] = (int)candidate;
                }
            }
        }

        memcpy(out + source * n, dist, n * sizeof(int));
    }

    free(dist);
    free(visited);
}
