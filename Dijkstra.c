#define _POSIX_C_SOURCE 200809L

#include "graph_utils.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

typedef struct {
    size_t size;
    double density;
    int max_weight;
    int infinity;
    uint64_t seed;
    int iterations;
    int quiet;
    int verify_with_floyd;
    int undirected;
    const char *graph_in;
    const char *graph_out;
    const char *csv_path;
} Options;

typedef struct {
    double elapsed_ms;
} RunResult;

static void print_usage(const char *program) {
    fprintf(stdout,
            "Usage: %s [options]\n"
            "\n"
            "Options:\n"
            "  -n, --size <value>           Number of vertices (default: 512)\n"
            "  -d, --density <0-1>          Edge density for random graphs (default: 0.2)\n"
            "  -w, --max-weight <value>     Maximum edge weight (default: 100)\n"
            "  -i, --infinity <value>       Value representing no edge (default: 999999)\n"
            "  -s, --seed <value>           RNG seed (default: 0xABCDEF)\n"
            "  -r, --iterations <value>     Number of repeated runs (default: 3)\n"
            "      --graph-in <file>        Load graph matrix from file instead of random generation\n"
            "      --graph-out <file>       Persist generated graph to file (after generation)\n"
            "      --csv <file>             Append run metrics to CSV file\n"
            "      --verify                 Cross-check results with Floyd-Warshall\n"
            "      --directed               Generate directed graphs (default)\n"
            "      --undirected             Generate undirected graphs\n"
            "      --quiet                  Suppress human-readable output\n"
            "      --verbose                Enable human-readable output (default)\n"
            "      --help                   Show this help message\n"
            "\n"
            "Examples:\n"
            "  %s --size 1024 --density 0.1 --verify\n"
            "  %s -n 2048 -r 5 --csv reports/dijkstra_results.csv --quiet\n",
            program, program, program);
}

static Options default_options(void) {
    Options opt = {
        .size = 512,
        .density = 0.2,
        .max_weight = 100,
        .infinity = GRAPH_INFINITY_DEFAULT,
        .seed = 0xABCDEF,
        .iterations = 3,
        .quiet = 0,
        .verify_with_floyd = 0,
        .undirected = 0,
        .graph_in = NULL,
        .graph_out = NULL,
        .csv_path = NULL,
    };
    return opt;
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
    while ((opt = getopt_long(argc, argv, "n:d:w:i:s:r:", long_opts, NULL)) != -1) {
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
                out->verify_with_floyd = 1;
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

    if (out->size == 0 || out->iterations <= 0) {
        fprintf(stderr, "Invalid configuration: size must be > 0 and iterations must be > 0\n");
        return -1;
    }

    return 0;
}

static double monotonic_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void floyd_warshall_reference(const MatrixGraph *graph, int *dist) {
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

static void dijkstra_all_pairs(const MatrixGraph *graph, int *out) {
    const size_t n = graph->order;
    const int inf = graph->infinity;
    int *dist = (int *)malloc(n * sizeof(int));
    unsigned char *visited = (unsigned char *)malloc(n);

    if (!dist || !visited) {
        fprintf(stderr, "Allocation failure in dijkstra_all_pairs\n");
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
                if (weight == inf) {
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

static int ensure_csv_header(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0) {
        return 0;
    }
    FILE *file = fopen(path, "w");
    if (!file) {
        fprintf(stderr, "Failed to open CSV file %s for writing header: %s\n", path, strerror(errno));
        return -1;
    }
    fprintf(file, "algorithm,size,density,max_weight,infinity,seed,iteration,time_ms\n");
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
    fprintf(file, "dijkstra-apsp,%zu,%.4f,%d,%d,%" PRIu64 ",%d,%.6f\n",
            opt->size,
            opt->density,
            opt->max_weight,
            opt->infinity,
            opt->seed,
            iteration,
            time_ms);
    fclose(file);
}

static void print_summary(const Options *opt, const RunResult *results, int count) {
    if (opt->quiet) {
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
    printf("Dijkstra APSP\n");
    printf("  size        : %zu\n", opt->size);
    printf("  density     : %.3f\n", opt->density);
    printf("  max weight  : %d\n", opt->max_weight);
    printf("  infinity    : %d\n", opt->infinity);
    printf("  seed        : %" PRIu64 "\n", opt->seed);
    printf("  graph type  : %s\n", opt->undirected ? "undirected" : "directed");
    printf("  iterations  : %d\n", count);
    printf("  time (ms)   : avg=%.3f min=%.3f max=%.3f\n", avg, min, max);
}

static int verify_results(const MatrixGraph *graph, const int *dijkstra_dist) {
    const size_t n = graph->order;
    int *reference = (int *)malloc(n * n * sizeof(int));
    if (!reference) {
        fprintf(stderr, "Failed to allocate memory for verification\n");
        return -1;
    }
    floyd_warshall_reference(graph, reference);

    int mismatches = 0;
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < n; ++col) {
            const size_t idx = matrix_index(graph, row, col);
            if (dijkstra_dist[idx] != reference[idx]) {
                ++mismatches;
            }
        }
    }

    free(reference);
    return mismatches;
}

int main(int argc, char **argv) {
    Options opt;
    const int parse_status = parse_args(argc, argv, &opt);
    if (parse_status != 0) {
        return parse_status > 0 ? 0 : 1;
    }

    MatrixGraph graph = {0};
    if (opt.graph_in) {
        read_graph_from_file(&graph, opt.graph_in);
        if (graph.order == 0) {
            fprintf(stderr, "Failed to load graph from %s\n", opt.graph_in);
            return 1;
        }
        if (opt.size != graph.order) {
            fprintf(stderr, "Overriding size %zu with graph size %zu from %s\n",
                    opt.size, graph.order, opt.graph_in);
            opt.size = graph.order;
        }
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
            fprintf(stderr, "Graph generation failed (out of memory?)\n");
            return 1;
        }
        if (opt.graph_out) {
            write_graph_to_file(&graph, opt.graph_out);
        }
    }

    const size_t n = graph.order;
    int *distances = (int *)malloc(n * n * sizeof(int));
    if (!distances) {
        fprintf(stderr, "Failed to allocate memory for APSP distances\n");
        free_matrix_graph(&graph);
        return 1;
    }

    RunResult *results = (RunResult *)calloc((size_t)opt.iterations, sizeof(RunResult));
    if (!results) {
        fprintf(stderr, "Allocation failure for results buffer\n");
        free(distances);
        free_matrix_graph(&graph);
        return 1;
    }

    for (int iter = 0; iter < opt.iterations; ++iter) {
        const double start_ms = monotonic_time_ms();
        dijkstra_all_pairs(&graph, distances);
        const double end_ms = monotonic_time_ms();
        const double elapsed = end_ms - start_ms;
        results[iter].elapsed_ms = elapsed;
        append_csv(&opt, iter, elapsed);
        if (opt.quiet && !opt.csv_path) {
            printf("dijkstra-apsp,size=%zu,iteration=%d,time_ms=%.6f\n", opt.size, iter, elapsed);
        }
    }

    if (opt.verify_with_floyd) {
        const int mismatches = verify_results(&graph, distances);
        if (mismatches < 0) {
            fprintf(stderr, "Verification skipped due to allocation error\n");
        } else if (mismatches == 0) {
            printf("Verification: PASS (Dijkstra matches Floyd-Warshall)\n");
        } else {
            printf("Verification: FAIL (%d mismatched entries)\n", mismatches);
        }
    }

    print_summary(&opt, results, opt.iterations);

    free(results);
    free(distances);
    free_matrix_graph(&graph);
    return 0;
}
