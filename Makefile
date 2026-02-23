# Project-wide build configuration
CC              := gcc
NVCC            := nvcc
CFLAGS          := -std=c11 -O3 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS         := -lm
# Target multiple architectures for broad GPU compatibility. Override for faster builds:
#   make gpu NVCC_ARCH="-gencode arch=compute_86,code=sm_86"
NVCC_ARCH      ?= -gencode arch=compute_50,code=sm_50 \
                  -gencode arch=compute_60,code=sm_60 \
                  -gencode arch=compute_70,code=sm_70 \
                  -gencode arch=compute_75,code=sm_75 \
                  -gencode arch=compute_86,code=sm_86 \
                  -gencode arch=compute_86,code=compute_86
NVCCFLAGS       := -O3 $(NVCC_ARCH) -Iinclude --compiler-options "-Wall -Wextra"

BIN_DIR         := bin
BUILD_DIR       := build
GRAPH_UTILS_OBJ := $(BUILD_DIR)/graph_utils.o
APSP_CPU_OBJ    := $(BUILD_DIR)/apsp_cpu.o
FLOYD_GPU_OBJ   := $(BUILD_DIR)/floyd_gpu_impl.o
DIJKSTRA_GPU_OBJ:= $(BUILD_DIR)/dijkstra_gpu_impl.o

CPU_FLOYD       := $(BIN_DIR)/floyd_cpu
CPU_DIJKSTRA    := $(BIN_DIR)/dijkstra_cpu
GPU_FLOYD       := $(BIN_DIR)/floyd_gpu
GPU_DIJKSTRA    := $(BIN_DIR)/dijkstra_gpu
VERIFICATION    := $(BIN_DIR)/verification

.PHONY: all cpu gpu verification clean

all: cpu gpu verification

cpu: $(CPU_FLOYD) $(CPU_DIJKSTRA)

gpu: $(GPU_FLOYD) $(GPU_DIJKSTRA)

verification: $(VERIFICATION)

$(BIN_DIR) $(BUILD_DIR):
	@mkdir -p $@

$(GRAPH_UTILS_OBJ): src/graph_utils.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(APSP_CPU_OBJ): src/apsp_cpu.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(FLOYD_GPU_OBJ): src/floyd_gpu_impl.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(DIJKSTRA_GPU_OBJ): src/dijkstra_gpu_impl.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(CPU_FLOYD): floyd.c $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) -o $@ $(LDFLAGS)

$(CPU_DIJKSTRA): Dijkstra.c $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) -o $@ $(LDFLAGS)

$(GPU_FLOYD): floyd.cu $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) -o $@

$(GPU_DIJKSTRA): Dijkstra.cu $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) -o $@

$(VERIFICATION): verification.cu $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) $(FLOYD_GPU_OBJ) $(DIJKSTRA_GPU_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(GRAPH_UTILS_OBJ) $(APSP_CPU_OBJ) $(FLOYD_GPU_OBJ) $(DIJKSTRA_GPU_OBJ) -o $@

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)
