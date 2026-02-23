#define _POSIX_C_SOURCE 200809L

#include "floyd_gpu.h"
#include "graph_utils.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOYD_CHECK(expr)                                                        \
    do {                                                                         \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__,   \
                    __LINE__, cudaGetErrorString(_err));                         \
            return;                                                              \
        }                                                                        \
    } while (0)

/* Phase 1: Pivot tile (K,K). k_block = K*B; kernel iterates [k_block, k_block+B) internally. */
__global__ __launch_bounds__(1024) void floyd_phase1_pivot(
    int *__restrict__ dist, int n, int k_block, int inf, int B) {
    __shared__ int tile[32][33];
    const int K = k_block / B;
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int g_row = K * B + row;
    const int g_col = K * B + col;
    int val = inf;
    if (g_row < n && g_col < n) val = dist[g_row * n + g_col];
    tile[row][col] = val;
    __syncthreads();
    const int limit = min(B, (int)(n - (size_t)K * B));
#pragma unroll 8
    for (int kk_local = 0; kk_local < limit; ++kk_local) {
        const int dik = tile[row][kk_local];
        const int dkj = tile[kk_local][col];
        if (dik != inf && dkj != inf) val = min(val, dik + dkj);
        __syncthreads();
        tile[row][col] = val;
        __syncthreads();
    }
    if (g_row < n && g_col < n) dist[g_row * n + g_col] = val;
}

__global__ __launch_bounds__(1024) void floyd_phase2_row_col(
    int *__restrict__ dist, int n, int k_block, int inf, int B) {
    const int K = k_block / B;
    const int tile_i = blockIdx.y;
    const int tile_j = blockIdx.x;
    if (tile_i != K && tile_j != K) return;
    if (tile_i == K && tile_j == K) return;
    __shared__ int pivot[32][33];
    __shared__ int own[32][33];
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int g_row = tile_i * B + row;
    const int g_col = tile_j * B + col;
    int p_val = inf;
    if (K * B + row < n && K * B + col < n)
        p_val = dist[(K * B + row) * n + (K * B + col)];
    pivot[row][col] = p_val;
    int o_val = inf;
    if (g_row < n && g_col < n) o_val = dist[g_row * n + g_col];
    own[row][col] = o_val;
    __syncthreads();
    const int limit = min(B, (int)(n - (size_t)K * B));
    const int is_row_block = (tile_i == K);
#pragma unroll 8
    for (int kk = 0; kk < limit; ++kk) {
        const int dik = is_row_block ? pivot[row][kk] : own[row][kk];
        const int dkj = is_row_block ? own[kk][col] : pivot[kk][col];
        if (dik != inf && dkj != inf) o_val = min(o_val, dik + dkj);
    }
    if (g_row < n && g_col < n) dist[g_row * n + g_col] = o_val;
}

__global__ __launch_bounds__(1024) void floyd_phase3_remaining(
    int *__restrict__ dist, int n, int k_block, int inf, int B) {
    const int K = k_block / B;
    const int tile_i = blockIdx.y;
    const int tile_j = blockIdx.x;
    if (tile_i == K || tile_j == K) return;
    __shared__ int row_tile[32][33];
    __shared__ int col_tile[32][33];
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int g_row = tile_i * B + row;
    const int g_col = tile_j * B + col;
    int r_val = inf, c_val = inf;
    if (tile_i * B + row < n && K * B + col < n)
        r_val = dist[(tile_i * B + row) * n + (K * B + col)];
    if (K * B + row < n && tile_j * B + col < n)
        c_val = dist[(K * B + row) * n + (tile_j * B + col)];
    row_tile[row][col] = r_val;
    col_tile[row][col] = c_val;
    int o_val = inf;
    if (g_row < n && g_col < n) o_val = dist[g_row * n + g_col];
    __syncthreads();
    const int limit = min(B, (int)(n - (size_t)K * B));
#define UF 4
    int cand[UF];
    for (int u = 0; u < UF; ++u) cand[u] = inf;
    int ki = 0;
    for (; ki + UF <= limit; ki += UF) {
        for (int u = 0; u < UF; ++u) {
            const int kk = ki + u;
            const int dik = row_tile[row][kk];
            const int dkj = col_tile[kk][col];
            if (dik != inf && dkj != inf) cand[u] = dik + dkj;
        }
        for (int u = 0; u < UF; ++u) { o_val = min(o_val, cand[u]); cand[u] = inf; }
    }
    for (; ki < limit; ++ki) {
        const int dik = row_tile[row][ki];
        const int dkj = col_tile[ki][col];
        if (dik != inf && dkj != inf) o_val = min(o_val, dik + dkj);
    }
#undef UF
    if (g_row < n && g_col < n) dist[g_row * n + g_col] = o_val;
}

void run_floyd_gpu(const MatrixGraph *graph, int *dist_out, int block_dim) {
    const size_t n = graph->order;
    const size_t bytes = n * n * sizeof(int);
    int *d_dist = NULL;
    FLOYD_CHECK(cudaMalloc((void **)&d_dist, bytes));
    FLOYD_CHECK(cudaMemcpy(d_dist, graph->weights, bytes, cudaMemcpyHostToDevice));
    const int B = block_dim;
    dim3 block(B, B);
    const int P = (int)((n + (size_t)B - 1) / B);
    dim3 grid2(P, P);
    for (int k_block = 0; k_block < (int)n; k_block += B) {
        floyd_phase1_pivot<<<1, block>>>(d_dist, (int)n, k_block, graph->infinity, B);
        FLOYD_CHECK(cudaGetLastError());
        floyd_phase2_row_col<<<grid2, block>>>(d_dist, (int)n, k_block, graph->infinity, B);
        FLOYD_CHECK(cudaGetLastError());
        floyd_phase3_remaining<<<grid2, block>>>(d_dist, (int)n, k_block, graph->infinity, B);
        FLOYD_CHECK(cudaGetLastError());
    }
    FLOYD_CHECK(cudaDeviceSynchronize());
    FLOYD_CHECK(cudaMemcpy(dist_out, d_dist, bytes, cudaMemcpyDeviceToHost));
    FLOYD_CHECK(cudaFree(d_dist));
}
