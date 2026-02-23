#define _POSIX_C_SOURCE 200809L

#include "apsp_cpu.h"
#include "floyd_gpu.h"
#include "dijkstra_gpu.h"
#include "graph_utils.h"
#include "verify_util.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { ALG_FLOYD, ALG_DIJKSTRA, ALG_BOTH } Algorithm;

typedef struct {
    size_t size;
    double density;
    int max_weight;
    int infinity;
    uint64_t seed;
    int block_dim;
    int undirected;
    const char *graph_in;
    Algorithm algorithm;
} VerifyOptions;

static VerifyOptions default_options(void) {
    VerifyOptions opt = {
        .size = 256,
        .density = 0.2,
        .max_weight = 100,
        .infinity = GRAPH_INFINITY_DEFAULT,
        .seed = 0xCAFEBABE,
        .block_dim = 32,
        .undirected = 1,
        .graph_in = NULL,
        .algorithm = ALG_BOTH,
    };
    return opt;
}

static void print_usage(const char *program) {
    printf("Usage: %s [options]\n\n", program);
    puts("Options:");
    puts("  -n, --size <value>           Number of vertices (default: 256)");
    puts("  -d, --density <0-1>          Edge density (default: 0.2)");
    puts("  -w, --max-weight <value>     Maximum edge weight (default: 100)");
    puts("  -i, --infinity <value>       Infinity value (default: 999999)");
    puts("  -s, --seed <value>           RNG seed (default: 0xCAFEBABE)");
    puts("  -b, --block <value>          Floyd block size (default: 32); Dijkstra uses 256");
    puts("      --graph-in <file>        Load graph from file instead of generating");
    puts("      --floyd                  Verify Floyd-Warshall only");
    puts("      --dijkstra               Verify Dijkstra only");
    puts("      --both                   Verify both algorithms (default)");
    puts("      --help                   Show this message");
}

static int parse_args(int argc, char **argv, VerifyOptions *out) {
    *out = default_options();

    const struct option long_opts[] = {
        {"size", required_argument, NULL, 'n'},
        {"density", required_argument, NULL, 'd'},
        {"max-weight", required_argument, NULL, 'w'},
        {"infinity", required_argument, NULL, 'i'},
        {"seed", required_argument, NULL, 's'},
        {"block", required_argument, NULL, 'b'},
        {"graph-in", required_argument, NULL, 1000},
        {"floyd", no_argument, NULL, 1001},
        {"dijkstra", no_argument, NULL, 1002},
        {"both", no_argument, NULL, 1003},
        {"help", no_argument, NULL, 1004},
        {0, 0, 0, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "n:d:w:i:s:b:", long_opts, NULL)) != -1) {
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
            case 'b':
                out->block_dim = (int)strtol(optarg, NULL, 10);
                break;
            case 1000:
                out->graph_in = optarg;
                break;
            case 1001:
                out->algorithm = ALG_FLOYD;
                break;
            case 1002:
                out->algorithm = ALG_DIJKSTRA;
                break;
            case 1003:
                out->algorithm = ALG_BOTH;
                break;
            case 1004:
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

    if (out->size == 0) {
        fprintf(stderr, "Invalid configuration: size must be > 0\n");
        return -1;
    }
    if (out->block_dim > 32 && out->algorithm != ALG_DIJKSTRA) {
        fprintf(stderr, "Floyd block size must be <= 32 (use --dijkstra for block=256)\n");
        return -1;
    }

    return 0;
}

static int do_verify_floyd(const MatrixGraph *graph, const VerifyOptions *opt) {
    const size_t n = graph->order;
    const size_t bytes = n * n * sizeof(int);

    int *cpu_result = (int *)malloc(bytes);
    int *gpu_result = (int *)malloc(bytes);
    if (!cpu_result || !gpu_result) {
        fprintf(stderr, "Allocation failure for Floyd verification\n");
        free(cpu_result);
        free(gpu_result);
        return -1;
    }

    printf("Running Floyd-Warshall CPU...\n");
    floyd_warshall_cpu(graph, cpu_result);

    printf("Running Floyd-Warshall GPU (block=%d)...\n", opt->block_dim);
    run_floyd_gpu(graph, gpu_result, opt->block_dim);

    const int mismatches = verify_matrices(cpu_result, gpu_result, n, 10);
    if (mismatches > 0) {
        printf("\n[Debug] First 5 entries: graph.weights vs gpu_result (row-major)\n");
        for (int i = 0; i < 5 && i < (int)(n * n); ++i) {
            printf("  [%d] weights=%d gpu_result=%d\n", i, graph->weights[i], gpu_result[i]);
        }
    }
    free(cpu_result);
    free(gpu_result);

    if (mismatches == 0) {
        printf("Floyd-Warshall: PASS (CPU and GPU match)\n");
        return 0;
    }
    printf("Floyd-Warshall: FAIL (%d mismatched entries, first 10 shown above)\n", mismatches);
    return 1;
}

static int do_verify_dijkstra(const MatrixGraph *graph, const VerifyOptions *opt) {
    const size_t n = graph->order;
    const size_t bytes = n * n * sizeof(int);

    int *cpu_result = (int *)malloc(bytes);
    int *gpu_result = (int *)malloc(bytes);
    if (!cpu_result || !gpu_result) {
        fprintf(stderr, "Allocation failure for Dijkstra verification\n");
        free(cpu_result);
        free(gpu_result);
        return -1;
    }

    printf("Running Dijkstra APSP CPU...\n");
    dijkstra_apsp_cpu(graph, cpu_result);

    printf("Running Dijkstra APSP GPU (block=%d)...\n", opt->block_dim);
    run_dijkstra_gpu(graph, gpu_result, opt->block_dim);

    const int mismatches = verify_matrices(cpu_result, gpu_result, n, 10);
    if (mismatches > 0) {
        printf("\n[Debug] First 5 entries: graph.weights vs gpu_result (row-major)\n");
        for (int i = 0; i < 5 && i < (int)(n * n); ++i) {
            printf("  [%d] weights=%d gpu_result=%d\n", i, graph->weights[i], gpu_result[i]);
        }
    }
    free(cpu_result);
    free(gpu_result);

    if (mismatches == 0) {
        printf("Dijkstra APSP: PASS (CPU and GPU match)\n");
        return 0;
    }
    printf("Dijkstra APSP: FAIL (%d mismatched entries, first 10 shown above)\n", mismatches);
    return 1;
}

int main(int argc, char **argv) {
    VerifyOptions opt;
    const int parse_status = parse_args(argc, argv, &opt);
    if (parse_status != 0) {
        return parse_status > 0 ? 0 : 1;
    }

    MatrixGraph graph = {0, NULL, 0, 0};
    if (opt.graph_in) {
        read_graph_from_file(&graph, opt.graph_in);
        if (graph.order == 0) {
            fprintf(stderr, "Failed to load graph from %s\n", opt.graph_in);
            return 1;
        }
        opt.size = graph.order;
    } else {
        GraphGenerationConfig cfg = {
            .order = opt.size,
            .density = opt.density,
            .max_weight = opt.max_weight,
            .infinity = opt.infinity,
            .seed = opt.seed,
            .undirected = opt.undirected,
        };
        graph = generate_random_matrix_graph(&cfg);
        if (graph.order == 0 || !graph.weights) {
            fprintf(stderr, "Graph generation failed\n");
            return 1;
        }
    }

    printf("Verification: n=%zu, density=%.2f\n", graph.order, opt.density);

    /* Floyd uses 2D block (BxB); Dijkstra uses 1D block of 256 threads. */
    int floyd_block = (opt.block_dim <= 32) ? opt.block_dim : 32;
    int dijkstra_block = 256; /* block-per-source kernel block size */

    VerifyOptions floyd_opt = opt;
    floyd_opt.block_dim = floyd_block;
    VerifyOptions dijkstra_opt = opt;
    dijkstra_opt.block_dim = dijkstra_block;

    int failed = 0;
    if (opt.algorithm == ALG_FLOYD || opt.algorithm == ALG_BOTH) {
        if (do_verify_floyd(&graph, &floyd_opt) != 0) {
            failed = 1;
        }
    }
    if (opt.algorithm == ALG_DIJKSTRA || opt.algorithm == ALG_BOTH) {
        if (do_verify_dijkstra(&graph, &dijkstra_opt) != 0) {
            failed = 1;
        }
    }

    free_matrix_graph(&graph);
    return failed ? 1 : 0;
}
