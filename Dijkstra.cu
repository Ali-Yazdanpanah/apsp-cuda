#define _POSIX_C_SOURCE 200809L

#include "graph_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define CUDA_CALL(expr)                                                           \
    do {                                                                          \
        cudaError_t _err = (expr);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__,    \
                    __LINE__, cudaGetErrorString(_err));                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

typedef struct {
    size_t size;
    double density;
    int max_weight;
    int infinity;
    uint64_t seed;
    int iterations;
    int block_dim;
    int quiet;
    int verify_with_cpu;
    int undirected;
    const char *graph_in;
    const char *graph_out;
    const char *csv_path;
} Options;

typedef struct {
    double elapsed_ms;
} RunResult;

__global__ void relax_neighbors_kernel(const int *weights,
                                       const int *dist_current,
                                       int *dist_next,
                                       int n,
                                       int source_vertex,
                                       int source_distance,
                                       int infinity) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) {
        return;
    }

    const int current = dist_current[v];
    if (source_distance >= infinity) {
        dist_next[v] = current;
        return;
    }

    const int w = weights[source_vertex * n + v];
    if (w >= infinity) {
        dist_next[v] = current;
        return;
    }

    const long candidate = (long)source_distance + (long)w;
    if (candidate < current) {
        dist_next[v] = (int)candidate;
    } else {
        dist_next[v] = current;
    }
}

static void dijkstra_reference(const MatrixGraph *graph, int *out) {
    const size_t n = graph->order;
    const int inf = graph->infinity;
    int *dist = (int *)malloc(n * sizeof(int));
    unsigned char *visited = (unsigned char *)malloc(n);
    if (!dist || !visited) {
        fprintf(stderr, "Reference Dijkstra allocation failed\n");
        free(dist);
        free(visited);
        return;
    }

    for (size_t src = 0; src < n; ++src) {
        for (size_t i = 0; i < n; ++i) {
            dist[i] = inf;
            visited[i] = 0;
        }
        dist[src] = 0;

        for (size_t iter = 0; iter < n; ++iter) {
            size_t u = SIZE_MAX;
            int best = inf;
            for (size_t v = 0; v < n; ++v) {
                if (!visited[v] && dist[v] < best) {
                    best = dist[v];
                    u = v;
                }
            }
            if (u == SIZE_MAX) {
                break;
            }
            visited[u] = 1;

            for (size_t v = 0; v < n; ++v) {
                const int weight = graph->weights[u * n + v];
                if (weight >= inf) {
                    continue;
                }
                const long candidate = (long)best + (long)weight;
                if (candidate < dist[v]) {
                    dist[v] = (int)candidate;
                }
            }
        }

        memcpy(out + src * n, dist, n * sizeof(int));
    }

    free(dist);
    free(visited);
}

static Options default_options(void) {
    Options opt;
    opt.size = 1024;
    opt.density = 0.15;
    opt.max_weight = 100;
    opt.infinity = GRAPH_INFINITY_DEFAULT;
    opt.seed = 0xD157AULL;
    opt.iterations = 3;
    opt.block_dim = 256;
    opt.quiet = 0;
    opt.verify_with_cpu = 0;
    opt.undirected = 0;
    opt.graph_in = NULL;
    opt.graph_out = NULL;
    opt.csv_path = NULL;
    return opt;
}

static void print_usage(const char *program) {
    printf("Usage: %s [options]\n\n", program);
    puts("Options:");
    puts("  -n, --size <value>           Number of vertices (default: 1024)");
    puts("  -d, --density <0-1>          Edge density (default: 0.15)");
    puts("  -w, --max-weight <value>     Maximum edge weight (default: 100)");
    puts("  -i, --infinity <value>       Infinity value (default: 999999)");
    puts("  -s, --seed <value>           RNG seed (default: 0xD157A)");
    puts("  -r, --iterations <value>     Number of repeated runs (default: 3)");
    puts("  -b, --block <value>          CUDA block size (default: 256 threads)");
    puts("      --graph-in <file>        Load graph matrix from file");
    puts("      --graph-out <file>       Persist generated graph to file");
    puts("      --csv <file>             Append metrics to CSV file");
    puts("      --verify                 Verify against CPU implementation");
    puts("      --directed               Generate directed graph (default)");
    puts("      --undirected             Generate undirected graph");
    puts("      --quiet                  Suppress human-readable summary");
    puts("      --verbose                Enable human-readable summary (default)");
    puts("      --help                   Show this message");
}

static int parse_args(int argc, char **argv, Options *out) {
    *out = default_options();

    const struct option long_opts[] = {
        {"size", required_argument, NULL, 'n'},
        {"density", required_argument, NULL, 'd'},
        {"max-weight", required_argument, NULL, 'w'},
        {"infinity", required_argument, NULL, 'i'},
        {"seed", required_argument, NULL, 's'},
        {"iterations", required_argument, NULL, 'r'},
        {"block", required_argument, NULL, 'b'},
        {"graph-in", required_argument, NULL, 1000},
        {"graph-out", required_argument, NULL, 1001},
        {"csv", required_argument, NULL, 1002},
        {"verify", no_argument, NULL, 1003},
        {"directed", no_argument, NULL, 1004},
        {"undirected", no_argument, NULL, 1005},
        {"quiet", no_argument, NULL, 1006},
        {"verbose", no_argument, NULL, 1007},
        {"help", no_argument, NULL, 1008},
        {0, 0, 0, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:d:w:i:s:r:b:", long_opts, NULL)) != -1) {
        switch (opt) {
            case 'n':
                out->size = (size_t)strtoull(optarg, NULL, 10);
                break;
            case 'd':
                out->density = strtod(optarg, NULL);
                break;
            case 'w':
                out->max_weight = (int)strtol(optarg, NULL, 10);
                break;
            case 'i':
                out->infinity = (int)strtol(optarg, NULL, 10);
                break;
            case 's':
                out->seed = (uint64_t)strtoull(optarg, NULL, 10);
                break;
            case 'r':
                out->iterations = (int)strtol(optarg, NULL, 10);
                break;
            case 'b':
                out->block_dim = (int)strtol(optarg, NULL, 10);
                break;
            case 1000:
                out->graph_in = optarg;
                break;
            case 1001:
                out->graph_out = optarg;
                break;
            case 1002:
                out->csv_path = optarg;
                break;
            case 1003:
                out->verify_with_cpu = 1;
                break;
            case 1004:
                out->undirected = 0;
                break;
            case 1005:
                out->undirected = 1;
                break;
            case 1006:
                out->quiet = 1;
                break;
            case 1007:
                out->quiet = 0;
                break;
            case 1008:
                print_usage(argv[0]);
                return 1;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    if (optind < argc) {
        out->size = (size_t)strtoull(argv[optind], NULL, 10);
    }

    if (out->size == 0 || out->iterations <= 0 || out->block_dim <= 0) {
        fprintf(stderr, "Invalid configuration provided\n");
        return -1;
    }

    return 0;
}

static int ensure_csv_header(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0) {
        return 0;
    }
    FILE *file = fopen(path, "w");
    if (!file) {
        fprintf(stderr, "Failed to open CSV file %s: %s\n", path, strerror(errno));
        return -1;
    }
    fprintf(file, "algorithm,size,density,max_weight,infinity,seed,block_dim,iteration,wall_time_ms\n");
    fclose(file);
    return 0;
}

static void append_csv(const Options *opt, int iteration, double time_ms) {
    if (!opt->csv_path) {
        return;
    }
    if (ensure_csv_header(opt->csv_path) != 0) {
        return;
    }
    FILE *file = fopen(opt->csv_path, "a");
    if (!file) {
        fprintf(stderr, "Failed to append to CSV file %s: %s\n", opt->csv_path, strerror(errno));
        return;
    }
    fprintf(file, "dijkstra-gpu,%zu,%.4f,%d,%d,%" PRIu64 ",%d,%d,%.6f\n",
            opt->size,
            opt->density,
            opt->max_weight,
            opt->infinity,
            opt->seed,
            opt->block_dim,
            iteration,
            time_ms);
    fclose(file);
}

static double monotonic_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void print_summary(const Options *opt, const RunResult *results, int count) {
    if (opt->quiet || count == 0) {
        return;
    }
    double total = 0.0;
    double min = results[0].elapsed_ms;
    double max = results[0].elapsed_ms;
    for (int i = 0; i < count; ++i) {
        const double t = results[i].elapsed_ms;
        total += t;
        if (t < min) min = t;
        if (t > max) max = t;
    }
    const double avg = total / count;

    printf("Dijkstra Hybrid CUDA APSP\n");
    printf("  size        : %zu\n", opt->size);
    printf("  density     : %.3f\n", opt->density);
    printf("  max weight  : %d\n", opt->max_weight);
    printf("  infinity    : %d\n", opt->infinity);
    printf("  seed        : %" PRIu64 "\n", opt->seed);
    printf("  graph type  : %s\n", opt->undirected ? "undirected" : "directed");
    printf("  block dim   : %d\n", opt->block_dim);
    printf("  iterations  : %d\n", count);
    printf("  wall time ms: avg=%.3f min=%.3f max=%.3f\n", avg, min, max);
}

int main(int argc, char **argv) {
    Options opt;
    const int parse_status = parse_args(argc, argv, &opt);
    if (parse_status != 0) {
        return parse_status > 0 ? 0 : 1;
    }

    MatrixGraph graph = {};
    if (opt.graph_in) {
        read_graph_from_file(&graph, opt.graph_in);
        if (graph.order == 0) {
            fprintf(stderr, "Failed to load graph from %s\n", opt.graph_in);
            return 1;
        }
        if (graph.order != opt.size) {
            fprintf(stderr, "Adjusting size %zu -> %zu (graph input)\n", opt.size, graph.order);
            opt.size = graph.order;
        }
    } else {
        GraphGenerationConfig cfg;
        cfg.order = opt.size;
        cfg.density = opt.density;
        cfg.max_weight = opt.max_weight;
        cfg.infinity = opt.infinity;
        cfg.seed = opt.seed;
        cfg.undirected = opt.undirected;
        graph = generate_random_matrix_graph(&cfg);
        if (graph.order == 0 || !graph.weights) {
            fprintf(stderr, "Graph generation failed\n");
            return 1;
        }
        if (opt.graph_out) {
            write_graph_to_file(&graph, opt.graph_out);
        }
    }

    const size_t n = graph.order;
    const size_t matrix_bytes = n * n * sizeof(int);
    const size_t row_bytes = n * sizeof(int);

    int *device_weights = NULL;
    CUDA_CALL(cudaMalloc((void **)&device_weights, matrix_bytes));
    CUDA_CALL(cudaMemcpy(device_weights, graph.weights, matrix_bytes, cudaMemcpyHostToDevice));

    int *device_dist_buffers[2] = {NULL, NULL};
    CUDA_CALL(cudaMalloc((void **)&device_dist_buffers[0], row_bytes));
    CUDA_CALL(cudaMalloc((void **)&device_dist_buffers[1], row_bytes));

    int *host_dist = (int *)malloc(row_bytes);
    if (!host_dist) {
        fprintf(stderr, "Host memory allocation failed\n");
        free(host_dist);
        CUDA_CALL(cudaFree(device_weights));
        CUDA_CALL(cudaFree(device_dist_buffers[0]));
        CUDA_CALL(cudaFree(device_dist_buffers[1]));
        free_matrix_graph(&graph);
        return 1;
    }
    unsigned char *visited = (unsigned char *)malloc(n);
    if (!visited) {
        fprintf(stderr, "Visited array allocation failed\n");
        free(host_dist);
        CUDA_CALL(cudaFree(device_weights));
        CUDA_CALL(cudaFree(device_dist_buffers[0]));
        CUDA_CALL(cudaFree(device_dist_buffers[1]));
        free_matrix_graph(&graph);
        return 1;
    }

    RunResult *results = (RunResult *)calloc((size_t)opt.iterations, sizeof(RunResult));
    if (!results) {
        fprintf(stderr, "Allocation failure for results buffer\n");
        free(visited);
        free(host_dist);
        CUDA_CALL(cudaFree(device_weights));
        CUDA_CALL(cudaFree(device_dist_buffers[0]));
        CUDA_CALL(cudaFree(device_dist_buffers[1]));
        free_matrix_graph(&graph);
        return 1;
    }

    int *output = (int *)malloc(matrix_bytes);
    if (!output) {
        fprintf(stderr, "Allocation failure for output distances\n");
        free(results);
        free(visited);
        free(host_dist);
        CUDA_CALL(cudaFree(device_weights));
        CUDA_CALL(cudaFree(device_dist_buffers[0]));
        CUDA_CALL(cudaFree(device_dist_buffers[1]));
        free_matrix_graph(&graph);
        return 1;
    }

    dim3 block(opt.block_dim);
    dim3 grid((unsigned int)((n + block.x - 1) / block.x));

    for (int iter = 0; iter < opt.iterations; ++iter) {
        const double start_ms = monotonic_time_ms();

        for (size_t src = 0; src < n; ++src) {
            for (size_t i = 0; i < n; ++i) {
                host_dist[i] = graph.infinity;
                visited[i] = 0;
            }
            host_dist[src] = 0;

            CUDA_CALL(cudaMemcpy(device_dist_buffers[0], host_dist, row_bytes, cudaMemcpyHostToDevice));
            int current_buffer = 0;

            for (size_t step = 0; step < n; ++step) {
                size_t u = SIZE_MAX;
                int best = graph.infinity;
                for (size_t v = 0; v < n; ++v) {
                    if (!visited[v] && host_dist[v] < best) {
                        best = host_dist[v];
                        u = v;
                    }
                }
                if (u == SIZE_MAX) {
                    break;
                }
                visited[u] = 1;

                const int source_distance = host_dist[u];
                const int next_buffer = 1 - current_buffer;
                relax_neighbors_kernel<<<grid, block>>>(device_weights,
                                                        device_dist_buffers[current_buffer],
                                                        device_dist_buffers[next_buffer],
                                                        (int)n,
                                                        (int)u,
                                                        source_distance,
                                                        graph.infinity);
                CUDA_CALL(cudaDeviceSynchronize());

                current_buffer = next_buffer;
                CUDA_CALL(cudaMemcpy(host_dist, device_dist_buffers[current_buffer], row_bytes, cudaMemcpyDeviceToHost));
            }

            memcpy(output + src * n, host_dist, row_bytes);
        }

        const double end_ms = monotonic_time_ms();
        const double elapsed = end_ms - start_ms;
        results[iter].elapsed_ms = elapsed;
        append_csv(&opt, iter, elapsed);
        if (opt.quiet && !opt.csv_path) {
            printf("dijkstra-gpu,size=%zu,iteration=%d,wall_time_ms=%.6f\n", n, iter, elapsed);
        }
    }

    if (opt.verify_with_cpu) {
        int *reference = (int *)malloc(matrix_bytes);
        if (!reference) {
            fprintf(stderr, "Verification skipped (allocation failure)\n");
        } else {
            dijkstra_reference(&graph, reference);
            int mismatches = 0;
            for (size_t idx = 0; idx < n * n; ++idx) {
                if (reference[idx] != output[idx]) {
                    ++mismatches;
                }
            }
            if (mismatches == 0) {
                printf("Verification: PASS (GPU hybrid matches CPU Dijkstra)\n");
            } else {
                printf("Verification: FAIL (%d mismatches)\n", mismatches);
            }
            free(reference);
        }
    }

    print_summary(&opt, results, opt.iterations);

    free(output);
    free(results);
    free(visited);
    free(host_dist);
    CUDA_CALL(cudaFree(device_weights));
    CUDA_CALL(cudaFree(device_dist_buffers[0]));
    CUDA_CALL(cudaFree(device_dist_buffers[1]));
    free_matrix_graph(&graph);
    return 0;
}
