#define _POSIX_C_SOURCE 200809L

#include "dijkstra_gpu.h"
#include "graph_utils.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define DIJKSTRA_CHECK(expr)                                                     \
    do {                                                                         \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__,   \
                    __LINE__, cudaGetErrorString(_err));                         \
            return;                                                              \
        }                                                                        \
    } while (0)

#define WARP_SIZE 32

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

static __device__ void block_reduce_min_pair(int *out_vertex, int *out_dist,
                                             int my_vertex, int my_dist) {
    __shared__ int s_vertex[32];
    __shared__ int s_dist[32];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    warp_reduce_min_pair(&my_dist, &my_vertex);
    if (lane == 0) {
        s_vertex[warp_id] = my_vertex;
        s_dist[warp_id] = my_dist;
    }
    __syncthreads();
    if (warp_id == 0) {
        my_vertex = (lane < blockDim.x / WARP_SIZE) ? s_vertex[lane] : -1;
        my_dist = (lane < blockDim.x / WARP_SIZE) ? s_dist[lane] : 0x7fffffff;
        if (lane >= blockDim.x / WARP_SIZE) my_dist = 0x7fffffff;
        warp_reduce_min_pair(&my_dist, &my_vertex);
        if (lane == 0) {
            *out_vertex = my_vertex;
            *out_dist = my_dist;
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(256, 4) void dijkstra_sssp_block(
    const int *__restrict__ weights, int *__restrict__ dist, int n, int inf, int base_source) {
    const int source = base_source + blockIdx.x;
    if (source >= n) return;
    int *my_dist = dist + (size_t)source * n;
    __shared__ unsigned char visited[4096];
    __shared__ int s_best_vertex;
    __shared__ int s_best_dist;
    for (int v = threadIdx.x; v < n; v += blockDim.x) {
        my_dist[v] = (v == source) ? 0 : inf;
        if (v < 4096) visited[v] = 0;
    }
    __syncthreads();
    for (int iter = 0; iter < n; ++iter) {
        int local_best_v = -1;
        int local_best_d = 0x7fffffff;
        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            if (v < 4096 && visited[v]) continue;
            if (v >= 4096) continue;
            int d = my_dist[v];
            if (d < local_best_d) { local_best_d = d; local_best_v = v; }
        }
        if (local_best_v < 0) local_best_d = 0x7fffffff;
        block_reduce_min_pair(&s_best_vertex, &s_best_dist, local_best_v, local_best_d);
        const int u = s_best_vertex;
        const int best = s_best_dist;
        if (u < 0 || best >= inf) break;
        if (u < 4096) visited[u] = 1;
        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            const int w = weights[(size_t)u * n + v];
            if (w >= inf) continue;
            const long candidate = (long)best + (long)w;
            if (candidate < my_dist[v]) my_dist[v] = (int)candidate;
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(256, 4) void dijkstra_sssp_block_large(
    const int *__restrict__ weights, int *__restrict__ dist,
    unsigned char *__restrict__ visited, int n, int inf, int base_source) {
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
            if (d < local_best_d) { local_best_d = d; local_best_v = v; }
        }
        if (local_best_v < 0) local_best_d = 0x7fffffff;
        block_reduce_min_pair(&s_best_vertex, &s_best_dist, local_best_v, local_best_d);
        const int u = s_best_vertex;
        const int best = s_best_dist;
        if (u < 0 || best >= inf) break;
        my_visited[u] = 1;
        for (int v = threadIdx.x; v < n; v += blockDim.x) {
            const int w = weights[(size_t)u * n + v];
            if (w >= inf) continue;
            const long candidate = (long)best + (long)w;
            if (candidate < my_dist[v]) my_dist[v] = (int)candidate;
        }
        __syncthreads();
    }
}

/* Expects dense n×n row-major matrix in graph->weights (weights[u*n+v] = edge u->v).
 * No CSR conversion needed; same format as MatrixGraph. */
void run_dijkstra_gpu(const MatrixGraph *graph, int *dist_out, int block_dim) {
    const size_t n = graph->order;
    const size_t matrix_bytes = n * n * sizeof(int);
    const size_t dist_bytes = matrix_bytes;
    int *d_weights = NULL;
    int *d_dist = NULL;
    DIJKSTRA_CHECK(cudaMalloc((void **)&d_weights, matrix_bytes));
    DIJKSTRA_CHECK(cudaMalloc((void **)&d_dist, dist_bytes));
    DIJKSTRA_CHECK(cudaMemcpy(d_weights, graph->weights, matrix_bytes, cudaMemcpyHostToDevice));
    const int block_size = block_dim;
    const int num_blocks = (int)n;
    int use_large = (n > 4096);
    unsigned char *d_visited = NULL;
    if (use_large) {
        DIJKSTRA_CHECK(cudaMalloc((void **)&d_visited, n * n * sizeof(unsigned char)));
    }
    if (use_large) {
        dijkstra_sssp_block_large<<<num_blocks, block_size>>>(
            d_weights, d_dist, d_visited, (int)n, graph->infinity, 0);
    } else {
        dijkstra_sssp_block<<<num_blocks, block_size>>>(
            d_weights, d_dist, (int)n, graph->infinity, 0);
    }
    DIJKSTRA_CHECK(cudaGetLastError());
    DIJKSTRA_CHECK(cudaDeviceSynchronize());
    DIJKSTRA_CHECK(cudaMemcpy(dist_out, d_dist, dist_bytes, cudaMemcpyDeviceToHost));
    if (d_visited) DIJKSTRA_CHECK(cudaFree(d_visited));
    DIJKSTRA_CHECK(cudaFree(d_dist));
    DIJKSTRA_CHECK(cudaFree(d_weights));
}
