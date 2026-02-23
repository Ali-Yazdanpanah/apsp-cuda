#define _POSIX_C_SOURCE 200809L

#include "apsp_cpu.h"
#include "graph_utils.h"
#include "verify_util.h"

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

#define checkCudaErrors(expr)                                                     \
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
    double effective_gbps;
    double gflops;
} RunResult;

#define WARP_SIZE 32

/* Warp-level min reduction for (dist, vertex) pair using shuffle.
 * All lanes participate; lane 0 gets the global min (dist, vertex) in the warp. */
static __device__ __forceinline__ void warp_reduce_min_pair(int *dist, int *vertex) {
    int my_d = *dist;
    int my_v = *vertex;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        int other_d = __shfl_down_sync(0xffffffffu, my_d, offset);
        int other_v = __shfl_down_sync(0xffffffffu, my_v, offset);
        if (other_d < my_d || (other_d == my_d && other_v < my_v)) {
            my_d = other_d;
            my_v = other_v;
        }
    }
    *dist = my_d;
    *vertex = my_v;
}

/* Block-level min-reduction using warp shuffles. Each warp reduces to lane 0,
 * then warp 0 reduces across warp results. Returns (best_vertex, best_dist). */
static __device__ void block_reduce_min_pair(int *out_vertex, int *out_dist,
                                             int my_vertex, int my_dist) {
    __shared__ int s_vertex[32]; /* Max warps per block */
    __shared__ int s_dist[32];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    warp_reduce_min_pair(&my_dist, &my_vertex);

    /* Lane 0 of each warp writes to shared memory */
    if (lane == 0) {
        s_vertex[warp_id] = my_vertex;
        s_dist[warp_id] = my_dist;
    }
    __syncthreads();

    /* Warp 0 reduces the per-warp results using shuffles */
    if (warp_id == 0) {
        my_vertex = (lane < blockDim.x / WARP_SIZE) ? s_vertex[lane] : -1;
        my_dist = (lane < blockDim.x / WARP_SIZE) ? s_dist[lane] : 0x7fffffff;
        if (lane >= blockDim.x / WARP_SIZE) {
            my_dist = 0x7fffffff; /* ensure non-participating lanes have inf */
        }
        warp_reduce_min_pair(&my_dist, &my_vertex);
        if (lane == 0) {
            *out_vertex = my_vertex;
            *out_dist = my_dist;
        }
    }
    __syncthreads();
}

/* Fully device-side Dijkstra: one block per source. Uses warp shuffles for
 * block-level min-reduction to find the next closest vertex. */
__global__ __launch_bounds__(256, 4) void dijkstra_sssp_block(
    const int *__restrict__ weights,
    int *__restrict__ dist,
    int n,
    int inf,
    int base_source) {
    const int source = base_source + blockIdx.x;
    if (source >= n) return;

    /* Each block writes to dist[source * n : (source+1) * n] */
    int *my_dist = dist + (size_t)source * n;

    __shared__ unsigned char visited[4096]; /* Max vertices; use dynamic if n > 4096 */
    __shared__ int s_best_vertex;
    __shared__ int s_best_dist;

    /* Initialize: dist[source]=0, rest=inf. visited=0 for all (source picked in 1st iter) */
    for (int v = threadIdx.x; v < n; v += blockDim.x) {
        my_dist[v] = (v == source) ? 0 : inf;
        if (v < 4096) visited[v] = 0;
    }
    __syncthreads();

    for (int iter = 0; iter < n; ++iter) {
        /* Each thread finds local min among unvisited vertices in its chunk */
        int local_best_v = -1;
        int local_best_d = 0x7fffffff;
        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            if (v < 4096 && visited[v]) continue;
            if (v >= 4096) {
                /* Fallback for n > 4096: use shared flag or __ldg. For simplicity
                 * we assume n <= 4096; else we'd need a different visited scheme. */
                continue;
            }
            int d = my_dist[v];
            if (d < local_best_d) {
                local_best_d = d;
                local_best_v = v;
            }
        }
        if (local_best_v < 0) local_best_d = 0x7fffffff;

        /* Block-level min-reduction via warp shuffles */
        block_reduce_min_pair(&s_best_vertex, &s_best_dist,
                              local_best_v, local_best_d);

        const int u = s_best_vertex;
        const int best = s_best_dist;
        if (u < 0 || best >= inf) break;

        if (u < 4096) visited[u] = 1;

        /* Relax: all threads relax edges from u to their assigned vertices */
        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            const int w = weights[(size_t)u * n + v];
            if (w >= inf) continue;
            const long candidate = (long)best + (long)w;
            if (candidate < my_dist[v]) {
                my_dist[v] = (int)candidate;
            }
        }
        __syncthreads();
    }
}

/* Variant for n > 4096: use global visited in a separate array. We'll allocate
 * visited as device memory of size n * num_sources, or use a single bitmap.
 * For now, restrict to n <= 4096 to keep shared memory simple. */
__global__ __launch_bounds__(256, 4) void dijkstra_sssp_block_large(
    const int *__restrict__ weights,
    int *__restrict__ dist,
    unsigned char *__restrict__ visited,
    int n,
    int inf,
    int base_source) {
    const int source = base_source + blockIdx.x;
    if (source >= n) return;

    int *my_dist = dist + (size_t)source * n;
    unsigned char *my_visited = visited + (size_t)source * n;

    __shared__ int s_best_vertex;
    __shared__ int s_best_dist;

    for (int v = threadIdx.x; v < n; v += blockDim.x) {
        my_dist[v] = (v == source) ? 0 : inf;
        my_visited[v] = 0;
    }
    __syncthreads();

    for (int iter = 0; iter < n; ++iter) {
        int local_best_v = -1;
        int local_best_d = 0x7fffffff;
        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            if (my_visited[v]) continue;
            int d = my_dist[v];
            if (d < local_best_d) {
                local_best_d = d;
                local_best_v = v;
            }
        }
        if (local_best_v < 0) local_best_d = 0x7fffffff;

        block_reduce_min_pair(&s_best_vertex, &s_best_dist,
                              local_best_v, local_best_d);

        const int u = s_best_vertex;
        const int best = s_best_dist;
        if (u < 0 || best >= inf) break;

        my_visited[u] = 1;

        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            const int w = weights[(size_t)u * n + v];
            if (w >= inf) continue;
            const long candidate = (long)best + (long)w;
            if (candidate < my_dist[v]) {
                my_dist[v] = (int)candidate;
            }
        }
        __syncthreads();
    }
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
    fprintf(file, "algorithm,size,density,max_weight,infinity,seed,block_dim,iteration,wall_time_ms,effective_gbps,gflops\n");
    fclose(file);
    return 0;
}

static void append_csv(const Options *opt, int iteration, const RunResult *r) {
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
    fprintf(file, "dijkstra-gpu,%zu,%.4f,%d,%d,%" PRIu64 ",%d,%d,%.6f,%.2f,%.2f\n",
            opt->size,
            opt->density,
            opt->max_weight,
            opt->infinity,
            opt->seed,
            opt->block_dim,
            iteration,
            r->elapsed_ms,
            r->effective_gbps,
            r->gflops);
    fclose(file);
}

static void print_summary(const Options *opt, const RunResult *results, int count) {
    if (opt->quiet || count == 0) {
        return;
    }
    double total = 0.0;
    double min_t = results[0].elapsed_ms;
    double max_t = results[0].elapsed_ms;
    double sum_gbps = 0.0;
    double sum_gflops = 0.0;
    for (int i = 0; i < count; ++i) {
        const double t = results[i].elapsed_ms;
        total += t;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
        sum_gbps += results[i].effective_gbps;
        sum_gflops += results[i].gflops;
    }
    const double avg = total / count;

    printf("Dijkstra Block-Per-Source CUDA APSP (Warp Shuffle Min-Reduction)\n");
    printf("  size        : %zu\n", opt->size);
    printf("  density     : %.3f\n", opt->density);
    printf("  max weight  : %d\n", opt->max_weight);
    printf("  infinity    : %d\n", opt->infinity);
    printf("  seed        : %" PRIu64 "\n", opt->seed);
    printf("  graph type  : %s\n", opt->undirected ? "undirected" : "directed");
    printf("  block dim   : %d\n", opt->block_dim);
    printf("  iterations  : %d\n", count);
    printf("  wall time ms: avg=%.3f min=%.3f max=%.3f\n", avg, min_t, max_t);
    printf("  effective GB/s : avg=%.2f\n", sum_gbps / count);
    printf("  GFLOPS      : avg=%.2f\n", sum_gflops / count);
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
    const size_t dist_bytes = n * n * sizeof(int);

    int *device_weights = NULL;
    checkCudaErrors(cudaMalloc((void **)&device_weights, matrix_bytes));
    checkCudaErrors(cudaMemcpy(device_weights, graph.weights, matrix_bytes, cudaMemcpyHostToDevice));

    int *device_dist = NULL;
    checkCudaErrors(cudaMalloc((void **)&device_dist, dist_bytes));

    int *output = (int *)malloc(dist_bytes);
    if (!output) {
        fprintf(stderr, "Allocation failure for output distances\n");
        checkCudaErrors(cudaFree(device_dist));
        checkCudaErrors(cudaFree(device_weights));
        free_matrix_graph(&graph);
        return 1;
    }

    RunResult *results = (RunResult *)calloc((size_t)opt.iterations, sizeof(RunResult));
    if (!results) {
        fprintf(stderr, "Allocation failure for results buffer\n");
        free(output);
        checkCudaErrors(cudaFree(device_dist));
        checkCudaErrors(cudaFree(device_weights));
        free_matrix_graph(&graph);
        return 1;
    }

    const int block_size = opt.block_dim;
    int num_blocks = (int)n;

    unsigned char *device_visited = NULL;
    int use_large_kernel = (n > 4096);
    if (use_large_kernel) {
        checkCudaErrors(cudaMalloc((void **)&device_visited, n * n * sizeof(unsigned char)));
    }

    /* Dijkstra per source: O(n²) FLOPs per source (min scan n + relax n per iter, n iters).
     * n sources => O(n³) FLOPs total.
     * Bytes: weights n² (read per source), dist n² read+write per source => ~3*n³*4 bytes. */
    const double flops_per_source = (double)n * (double)n * 3.0; /* min scan + relax per iter */
    const double flops_total = flops_per_source * (double)n;
    const double bytes_total = 12.0 * (double)n * (double)n * (double)n; /* ~3*n³ ints */

    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    for (int iter = 0; iter < opt.iterations; ++iter) {
        checkCudaErrors(cudaEventRecord(start_event, 0));

        if (use_large_kernel) {
            dijkstra_sssp_block_large<<<num_blocks, block_size>>>(
                device_weights, device_dist, device_visited,
                (int)n, graph.infinity, 0);
        } else {
            dijkstra_sssp_block<<<num_blocks, block_size>>>(
                device_weights, device_dist, (int)n, graph.infinity, 0);
        }
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaEventRecord(stop_event, 0));
        checkCudaErrors(cudaEventSynchronize(stop_event));
        checkCudaErrors(cudaDeviceSynchronize());

        float elapsed_ms = 0.0f;
        checkCudaErrors(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        const double elapsed_sec = (double)elapsed_ms / 1000.0;
        results[iter].elapsed_ms = (double)elapsed_ms;
        results[iter].effective_gbps = (elapsed_sec > 0.0) ? (bytes_total / 1e9 / elapsed_sec) : 0.0;
        results[iter].gflops = (elapsed_sec > 0.0) ? (flops_total / 1e9 / elapsed_sec) : 0.0;
        append_csv(&opt, iter, &results[iter]);
        if (opt.quiet && !opt.csv_path) {
            printf("dijkstra-gpu,size=%zu,iteration=%d,wall_time_ms=%.6f,gbps=%.2f,gflops=%.2f\n",
                   n, iter, elapsed_ms, results[iter].effective_gbps, results[iter].gflops);
        }
    }

    checkCudaErrors(cudaMemcpy(output, device_dist, dist_bytes, cudaMemcpyDeviceToHost));

    if (opt.verify_with_cpu) {
        int *reference = (int *)malloc(dist_bytes);
        if (!reference) {
            fprintf(stderr, "Verification skipped (allocation failure)\n");
        } else {
            dijkstra_apsp_cpu(&graph, reference);
            const int mismatches = verify_matrices(reference, output, n, 10);
            if (mismatches == 0) {
                printf("Verification: PASS (GPU block-per-source matches CPU Dijkstra)\n");
            } else {
                printf("Verification: FAIL (%d mismatches, first 10 shown above)\n", mismatches);
            }
            free(reference);
        }
    }

    print_summary(&opt, results, opt.iterations);

    free(results);
    free(output);
    if (device_visited) {
        checkCudaErrors(cudaFree(device_visited));
    }
    checkCudaErrors(cudaFree(device_dist));
    checkCudaErrors(cudaFree(device_weights));
    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));
    free_matrix_graph(&graph);
    return 0;
}
