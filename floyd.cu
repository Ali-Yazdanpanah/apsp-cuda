#define _POSIX_C_SOURCE 200809L

#include "apsp_cpu.h"
#include "graph_utils.h"
#include "verify_util.h"
#include "floyd_gpu.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
    const char *graph_in;
    const char *graph_out;
    const char *csv_path;
} Options;

typedef struct {
    double elapsed_ms;
    double effective_gbps;
    double gflops;
} RunResult;

/* Phase 1: Update pivot tile (K,K) using only itself. Single block.
 * k_block is the start of the pivot block (K*B); kernel iterates over [k_block, k_block+B). */
}

static Options default_options(void) {
    Options opt;
    opt.size = 1024;
    opt.density = 0.3;
    opt.max_weight = 100;
    opt.infinity = GRAPH_INFINITY_DEFAULT;
    opt.seed = 0xF10A7ULL;
    opt.iterations = 3;
    opt.block_dim = 32;
    opt.quiet = 0;
    opt.verify_with_cpu = 0;
    opt.graph_in = NULL;
    opt.graph_out = NULL;
    opt.csv_path = NULL;
    return opt;
}

static void print_usage(const char *program) {
    printf("Usage: %s [options]\n\n", program);
    puts("Options:");
    puts("  -n, --size <value>           Number of vertices (default: 1024)");
    puts("  -d, --density <0-1>          Edge density for random graphs (default: 0.3)");
    puts("  -w, --max-weight <value>     Maximum edge weight (default: 100)");
    puts("  -i, --infinity <value>       Infinity value (default: 999999)");
    puts("  -s, --seed <value>           RNG seed (default: 0xF10A7)");
    puts("  -r, --iterations <value>     Number of repeated runs (default: 3)");
    puts("  -b, --block <value>          CUDA block size per dimension (default: 32)");
    puts("      --graph-in <file>        Load graph matrix from file");
    puts("      --graph-out <file>       Persist generated graph to file");
    puts("      --csv <file>             Append results to CSV file");
    puts("      --verify                 Verify with CPU Floyd-Warshall");
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
        {"quiet", no_argument, NULL, 1004},
        {"verbose", no_argument, NULL, 1005},
        {"help", no_argument, NULL, 1006},
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
                out->quiet = 1;
                break;
            case 1005:
                out->quiet = 0;
                break;
            case 1006:
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
    if (out->block_dim > 32) {
        fprintf(stderr, "Block size must be <= 32 (requested %d)\n", out->block_dim);
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
    fprintf(file, "algorithm,size,density,max_weight,infinity,seed,block_dim,iteration,gpu_time_ms,effective_gbps,gflops\n");
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
    fprintf(file, "floyd-warshall-gpu,%zu,%.4f,%d,%d,%" PRIu64 ",%d,%d,%.6f,%.2f,%.2f\n",
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

    printf("Floyd-Warshall CUDA APSP (Blocked 3-Phase)\n");
    printf("  size        : %zu\n", opt->size);
    printf("  density     : %.3f\n", opt->density);
    printf("  max weight  : %d\n", opt->max_weight);
    printf("  infinity    : %d\n", opt->infinity);
    printf("  seed        : %" PRIu64 "\n", opt->seed);
    printf("  block dim   : %d\n", opt->block_dim);
    printf("  iterations  : %d\n", count);
    printf("  GPU time ms : avg=%.3f min=%.3f max=%.3f\n", avg, min_t, max_t);
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
        cfg.undirected = 1;
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
    const size_t elements = n * n;
    const size_t bytes = elements * sizeof(int);

    int *device_dist = NULL;
    checkCudaErrors(cudaMalloc((void **)&device_dist, bytes));

    int *host_output = (int *)malloc(bytes);
    if (!host_output) {
        fprintf(stderr, "Allocation failure for host_output\n");
        checkCudaErrors(cudaFree(device_dist));
        free_matrix_graph(&graph);
        return 1;
    }

    RunResult *results = (RunResult *)calloc((size_t)opt.iterations, sizeof(RunResult));
    if (!results) {
        fprintf(stderr, "Allocation failure for results buffer\n");
        free(host_output);
        checkCudaErrors(cudaFree(device_dist));
        free_matrix_graph(&graph);
        return 1;
    }

    const int B = opt.block_dim;
    dim3 block(B, B);

    /* Phase 1: 1 block. Phase 2: row/col tiles. Phase 3: remaining tiles. */
    const int P = (int)((n + (size_t)B - 1) / B);

    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    int *reference = NULL;
    if (opt.verify_with_cpu) {
        reference = (int *)malloc(bytes);
        if (!reference) {
            fprintf(stderr, "Allocation failure for CPU verification buffer\n");
            free(results);
            free(host_output);
            checkCudaErrors(cudaFree(device_dist));
            free_matrix_graph(&graph);
            return 1;
        }
    }

    /* Floyd-Warshall: ~3 FLOPs per cell update per k, n² cells, n k-iterations => 3*n³ FLOPs */
    /* Bytes: 2 reads + 1 write per cell per k => 12*n³ bytes */
    const double flops_total = 3.0 * (double)n * (double)n * (double)n;
    const double bytes_total = 12.0 * (double)n * (double)n * (double)n;

    for (int iter = 0; iter < opt.iterations; ++iter) {
        checkCudaErrors(cudaMemcpy(device_dist, graph.weights, bytes, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaEventRecord(start_event, 0));
        /* Outer loop: one iteration per pivot BLOCK to cut host launch overhead from n to ceil(n/B) */
        dim3 grid2(P, P);
        for (int k_block = 0; k_block < (int)n; k_block += B) {
            floyd_phase1_pivot<<<1, block>>>(device_dist, (int)n, k_block, graph.infinity, B);
            checkCudaErrors(cudaGetLastError());
            floyd_phase2_row_col<<<grid2, block>>>(device_dist, (int)n, k_block, graph.infinity, B);
            checkCudaErrors(cudaGetLastError());
            floyd_phase3_remaining<<<grid2, block>>>(device_dist, (int)n, k_block, graph.infinity, B);
            checkCudaErrors(cudaGetLastError());
        }
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
            printf("floyd-warshall-gpu,size=%zu,iteration=%d,gpu_time_ms=%.6f,gbps=%.2f,gflops=%.2f\n",
                   n, iter, elapsed_ms, results[iter].effective_gbps, results[iter].gflops);
        }

        checkCudaErrors(cudaMemcpy(host_output, device_dist, bytes, cudaMemcpyDeviceToHost));
    }

    if (opt.verify_with_cpu) {
        floyd_warshall_cpu(&graph, reference);
        const int mismatches = verify_matrices(reference, host_output, n, 10);
        if (mismatches == 0) {
            printf("Verification: PASS (GPU matches CPU reference)\n");
        } else {
            printf("Verification: FAIL (%d mismatched entries, first 10 shown above)\n", mismatches);
        }
    }

    print_summary(&opt, results, opt.iterations);

    free(reference);
    checkCudaErrors(cudaEventDestroy(start_event));
    checkCudaErrors(cudaEventDestroy(stop_event));
    free(results);
    free(host_output);
    checkCudaErrors(cudaFree(device_dist));
    free_matrix_graph(&graph);
    return 0;
}
