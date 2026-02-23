# APSP Kernel Design — Graduate-Level HPC Refactor

This document outlines the proposed kernel signatures and design for the blocked Floyd–Warshall and fully device-side Dijkstra implementations. Implementations should follow these contracts.

---

## 1. Blocked Floyd–Warshall (Three-Phase)

The naive kernel issues one global memory read per `(i,k)` and `(k,j)` per thread per k-iteration. The blocked algorithm divides the N×N matrix into B×B tiles, caches tiles in shared memory, and requires **three dependent kernel launches per k-iteration** to maintain data consistency across tiles.

### Tile Layout

- **B** = block/tile size (e.g. 16 or 32)
- **P** = `ceil(n / B)` = number of tiles per dimension
- Pivot tile for round k: **(K, K)** where `K = k / B`
- Matrix stored row-major: `dist[i * n + j]`

### Phase 1: Pivot Tile

The pivot tile `(K,K)` must be fully updated using **only itself** before any other tile reads it. This is the tile containing diagonal element `(k,k)`. Internal Floyd–Warshall is run on this tile.

| Kernel | Purpose |
|--------|---------|
| `floyd_phase1_pivot` | Load pivot tile into shared memory, run B inner iterations (or `min(B, n - K*B)`), write back. Single block. |

**Proposed signature:**

```cuda
__global__ __launch_bounds__(BLOCK_DIM * BLOCK_DIM, 4)
void floyd_phase1_pivot(int *__restrict__ dist,
                        int n,
                        int k,
                        int inf,
                        int tile_size);
```

- **dist**: Global distance matrix (row-major, n×n)
- **n**: Matrix dimension
- **k**: Current pivot vertex index (0..n-1)
- **inf**: Infinity sentinel
- **tile_size**: B
- **Grid**: 1 block
- **Block**: `(tile_size, tile_size)` threads
- **Shared memory**: `(__shared__ int)[tile_size][tile_size]` for the pivot tile

### Phase 2: Row and Column Tiles

Tiles in the pivot row `(K, J)` and pivot column `(I, K)` for `J ≠ K` and `I ≠ K`. Each tile depends on the pivot tile. All such tiles can be updated in parallel (no tile reads another row/col tile).

| Kernel | Purpose |
|--------|---------|
| `floyd_phase2_row_col` | Each block handles one tile (I,J) where `I == K || J == K` and `(I,J) != (K,K)`. Load pivot row/col and own tile into shared memory; update; write back. |

**Proposed signature:**

```cuda
__global__ __launch_bounds__(BLOCK_DIM * BLOCK_DIM, 4)
void floyd_phase2_row_col(int *__restrict__ dist,
                          int n,
                          int k,
                          int inf,
                          int tile_size);
```

- **Grid**: 2D grid over tile indices; each block corresponds to one tile `(tile_i, tile_j)` where `tile_i == K || tile_j == K`, excluding `(K,K)`
- **Block**: `(tile_size, tile_size)` threads
- **Shared memory**:
  - Pivot row strip: `[tile_size]` (row K of pivot block, or column strip for col tiles)
  - Pivot column strip: `[tile_size]`
  - Own tile: `[tile_size][tile_size]`

### Phase 3: Remaining Tiles

All tiles `(I,J)` where `I ≠ K` and `J ≠ K`. Each needs the row tile `(I,K)` and column tile `(K,J)` plus its own data.

| Kernel | Purpose |
|--------|---------|
| `floyd_phase3_remaining` | Each block handles one tile (I,J) with `I != K` and `J != K`. Load row tile (I,K), col tile (K,J), own tile (I,J); update via pivot path; write back. |

**Proposed signature:**

```cuda
__global__ __launch_bounds__(BLOCK_DIM * BLOCK_DIM, 4)
void floyd_phase3_remaining(int *__restrict__ dist,
                            int n,
                            int k,
                            int inf,
                            int tile_size);
```

- **Grid**: 2D over tiles `(I,J)` where `I != K` and `J != K`
- **Block**: `(tile_size, tile_size)`
- **Shared memory**:
  - Row tile strip from `(I,K)`: `[tile_size][tile_size]` (or row of it)
  - Col tile strip from `(K,J)`: `[tile_size][tile_size]` (or col of it)
  - Own tile: `[tile_size][tile_size]`

### Host-Side Loop (per k-iteration)

```c
for (int k = 0; k < n; ++k) {
    floyd_phase1_pivot<<<1, block2d>>>(dist, n, k, inf, B);
    floyd_phase2_row_col<<<grid2_rc, block2d>>>(dist, n, k, inf, B);
    floyd_phase3_remaining<<<grid3_rem, block2d>>>(dist, n, k, inf, B);
}
```

### Non-Multiple Handling

When `n % B != 0`:

- Tiles on the right and bottom edges are partial.
- Use bounds checks: `if (row < n && col < n)` before reads/writes.
- Use sentinel values (e.g. `inf`) for out-of-bounds tile elements in shared memory to avoid incorrect updates.

---

## 2. Work-Efficient Parallel Dijkstra (Fully Device-Side)

Eliminate host-side vertex selection and all CPU–GPU synchronization during the N parallel source searches. Use a single kernel per source (or batched) with an internal loop over relaxation steps.

### Strategy: Block-Per-Source with Internal Min-Reduction

Each block runs one complete Dijkstra for one source. Within the block:

1. **Find min**: Parallel min-reduction over unvisited vertices to get `(u, dist[u])`.
2. **Relax**: All threads participate in relaxing edges from u to all vertices.
3. **Mark visited**: Set `visited[u] = 1`.
4. Repeat until no reachable unvisited vertex.

### Kernel 1: Single-Source Dijkstra (One Block Per Source)

| Kernel | Purpose |
|--------|---------|
| `dijkstra_sssp_block` | One block handles one source. Internal loop: block-level min-reduction → relax from min vertex → mark visited. No host sync. |

**Proposed signature:**

```cuda
__global__ __launch_bounds__(MAX_BLOCK_SIZE, 4)
void dijkstra_sssp_block(const int *__restrict__ weights,
                         int *__restrict__ dist,
                         int n,
                         int inf,
                         int base_source);
```

- **weights**: n×n adjacency matrix (row-major)
- **dist**: Output buffer of size `n * num_sources`; block `b` writes to `dist + b * n`
- **n**: Number of vertices
- **inf**: Infinity sentinel
- **base_source**: First source index for this grid (for batched launch: block `b` runs source `base_source + b`)
- **Grid**: 1D, one block per source (or batch)
- **Block**: 1D, 256–1024 threads (enough for coalesced relax; min-reduction uses shared memory)

### Alternative: Near-Far Binning

If strict Dijkstra ordering is relaxed for more parallelism, a **binning** strategy can be used:

| Kernel | Purpose |
|--------|---------|
| `dijkstra_bin_relax` | Process one "bin" (distance bucket). All vertices in the current bin relax their neighbors in parallel. Use atomics or deterministic merge for `dist` updates. |
| `dijkstra_bin_fill` | Populate the next bin from relaxed distances. |

This is closer to Δ-stepping. For strict Dijkstra, the block-per-source with internal min-reduction is the preferred design.

### Convergence / Global Flag (Optional)

To avoid iterating all n steps when the graph is sparse:

- Use a `__device__` flag or atomic: `converged`
- After relax, if no `dist` value changed, set `converged = 1`
- Thread 0 in the block checks and breaks the loop

Alternatively, track "frontier" size with a block-level count.

---

## 3. Profiling & Metrics (Host-Side)

Extend logging to capture:

| Metric | Formula |
|--------|---------|
| **Effective Bandwidth (GB/s)** | `(bytes_read + bytes_written) / (time_sec * 1e9)` |
| **GFLOPS** | `(flops_total) / (time_sec * 1e9)` |

### Floyd–Warshall FLOP and Byte Count

- **FLOPs per k**: ~3 per cell update (add, compare, optional assign) × n² ≈ 3n²
- **Total FLOPs**: 3 · n³
- **Bytes**: 2 reads + 1 write per cell per k ≈ 12 bytes per cell per k → 12 · n³

### Dijkstra (per source)

- **FLOPs**: O(n²) for dense (each of n iterations scans n vertices for min + relax)
- **Bytes**: Read `dist`, `weights`; write `dist`

### Proposed RunResult Extension

```c
typedef struct {
    double elapsed_ms;
    double effective_gbps;   // (bytes_read + bytes_written) / (elapsed_sec * 1e9)
    double gflops;           // flops / (elapsed_sec * 1e9)
} RunResult;
```

### CSV Extension

Append columns: `effective_gbps,gflops`

---

## 4. Optimizations Checklist

| Optimization | Floyd | Dijkstra |
|--------------|-------|----------|
| **`__restrict__`** on pointers | ✓ | ✓ |
| **`__launch_bounds__(max_threads, min_blocks_per_sm)`** | ✓ | ✓ |
| **Coalesced global access** | Row-major; tiles loaded in coalesced fashion | `dist[v]`, `weights[u*n+v]` by thread v |
| **Shared memory for hot data** | Pivot, row, col tiles | `dist` block-cached during reduction; relax reads coalesced |
| **Avoid bank conflicts** | Pad shared arrays if needed (e.g. `[][T+1]`) | — |
| **Block size** | 16×16 or 32×32 (balance occupancy vs shared mem) | 256–512 |

---

## 5. File Structure After Refactor

```
apsp-cuda/
├── include/
│   ├── graph_utils.h
│   └── apsp_kernels.h      # Kernel declarations (optional, if extracted)
├── src/
│   ├── graph_utils.c
│   ├── floyd_blocked.cu    # Phase 1, 2, 3 kernels + host loop
│   ├── dijkstra_device.cu  # Block-per-source kernel
│   └── metrics.h           # RunResult, effective_gbps, gflops helpers
├── floyd.cu                # Main: CLI, graph load, call floyd_blocked
├── Dijkstra.cu             # Main: CLI, graph load, call dijkstra_device
└── docs/
    └── KERNEL_DESIGN.md    # This file
```

---

## 6. Summary of Kernel Signatures

### Floyd–Warshall (3 kernels per k-iteration)

```cuda
__global__ __launch_bounds__(B*B, 4)
void floyd_phase1_pivot(int *__restrict__ dist, int n, int k, int inf, int B);

__global__ __launch_bounds__(B*B, 4)
void floyd_phase2_row_col(int *__restrict__ dist, int n, int k, int inf, int B);

__global__ __launch_bounds__(B*B, 4)
void floyd_phase3_remaining(int *__restrict__ dist, int n, int k, int inf, int B);
```

### Dijkstra (1 kernel, N blocks for N sources)

```cuda
__global__ __launch_bounds__(256, 4)
void dijkstra_sssp_block(const int *__restrict__ weights,
                         int *__restrict__ dist,
                         int n,
                         int inf,
                         int base_source);
```

---

*Next step: Implement these kernels in `floyd.cu` and `Dijkstra.cu` (or in the new `*_blocked.cu` / `*_device.cu` files), then wire up the metrics and CSV logging.*
