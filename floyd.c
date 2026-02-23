#define _POSIX_C_SOURCE 200809L

#include "apsp_cpu.h"
#include "graph_utils.h"

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
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
            "  -d, --density <0-1>          Edge density for random graphs (default: 0.3)\n"
            "  -w, --max-weight <value>     Maximum edge weight (default: 100)\n"
            "  -i, --infinity <value>       Value representing no edge (default: 999999)\n"
            "  -s, --seed <value>           RNG seed (default: 0xC0FFEE)\n"
            "  -r, --iterations <value>     Number of repeated runs (default: 3)\n"
            "      --graph-in <file>        Load graph matrix from file instead of random generation\n"
            "      --graph-out <file>       Persist generated graph to file (after generation)\n"
            "      --csv <file>             Append run metrics to CSV file\n"
            "      --quiet                  Only print metrics in CSV format\n"
            "      --verbose                Print additional information\n"
            "      --help                   Show this help message\n"
            "\n"
            "Examples:\n"
            "  %s --size 1024 --density 0.2 --seed 42\n"
            "  %s -n 2048 -r 5 --csv reports/floyd_results.csv\n",
            program, program, program);
}

static Options default_options(void) {
    Options opt = {
        .size = 512,
        .density = 0.3,
        .max_weight = 100,
        .infinity = GRAPH_INFINITY_DEFAULT,
        .seed = 0xC0FFEE,
        .iterations = 3,
        .quiet = 0,
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
        {"quiet", no_argument, NULL, 1003},
        {"verbose", no_argument, NULL, 1004},
        {"help", no_argument, NULL, 1005},
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
                out->quiet = 1;
                break;
            case 1004:
                out->quiet = 0;
                break;
            case 1005:
                print_usage(argv[0]);
                return 1;
            default:
                print_usage(argv[0]);
                return -1;
        }
    }

    if (optind < argc) {
        /* Backwards compatibility: allow passing size as positional argument. */
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

static void floyd_warshall(const MatrixGraph *graph, int *dist) {
    floyd_warshall_cpu(graph, dist);
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
    fprintf(file, "floyd-warshall,%zu,%.4f,%d,%d,%" PRIu64 ",%d,%.6f\n",
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

    printf("Floyd-Warshall APSP\n");
    printf("  size        : %zu\n", opt->size);
    printf("  density     : %.3f\n", opt->density);
    printf("  max weight  : %d\n", opt->max_weight);
    printf("  infinity    : %d\n", opt->infinity);
    printf("  seed        : %" PRIu64 "\n", opt->seed);
    printf("  iterations  : %d\n", count);
    printf("  time (ms)   : avg=%.3f min=%.3f max=%.3f\n", avg, min, max);
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
            .undirected = 1,
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
        fprintf(stderr, "Allocation failure for distance matrix\n");
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
        floyd_warshall(&graph, distances);
        const double end_ms = monotonic_time_ms();
        const double elapsed = end_ms - start_ms;
        results[iter].elapsed_ms = elapsed;
        append_csv(&opt, iter, elapsed);
        if (opt.quiet && !opt.csv_path) {
            /* When quiet is enabled without CSV we still output structured data. */
            printf("floyd-warshall,size=%zu,iteration=%d,time_ms=%.6f\n", opt.size, iter, elapsed);
        }
    }

    print_summary(&opt, results, opt.iterations);

    free(results);
    free(distances);
    free_matrix_graph(&graph);
    return 0;
}
