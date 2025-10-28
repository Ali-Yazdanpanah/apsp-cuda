# Project-wide build configuration
CC              := gcc
NVCC            := nvcc
CFLAGS          := -std=c11 -O3 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS         := -lm
NVCC_ARCH      ?= -gencode arch=compute_60,code=sm_60
NVCCFLAGS       := -O3 $(NVCC_ARCH) -Iinclude --compiler-options "-Wall -Wextra"

BIN_DIR         := bin
BUILD_DIR       := build
GRAPH_UTILS_SRC := src/graph_utils.c
GRAPH_UTILS_OBJ := $(BUILD_DIR)/graph_utils.o

CPU_FLOYD       := $(BIN_DIR)/floyd_cpu
CPU_DIJKSTRA    := $(BIN_DIR)/dijkstra_cpu
GPU_FLOYD       := $(BIN_DIR)/floyd_gpu
GPU_DIJKSTRA    := $(BIN_DIR)/dijkstra_gpu

.PHONY: all cpu gpu clean

all: cpu gpu

cpu: $(CPU_FLOYD) $(CPU_DIJKSTRA)

gpu: $(GPU_FLOYD) $(GPU_DIJKSTRA)

$(BIN_DIR) $(BUILD_DIR):
	@mkdir -p $@

$(GRAPH_UTILS_OBJ): $(GRAPH_UTILS_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(CPU_FLOYD): floyd.c $(GRAPH_UTILS_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(GRAPH_UTILS_OBJ) -o $@ $(LDFLAGS)

$(CPU_DIJKSTRA): Dijkstra.c $(GRAPH_UTILS_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(GRAPH_UTILS_OBJ) -o $@ $(LDFLAGS)

$(GPU_FLOYD): floyd.cu $(GRAPH_UTILS_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(GRAPH_UTILS_OBJ) -o $@

$(GPU_DIJKSTRA): Dijkstra.cu $(GRAPH_UTILS_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< $(GRAPH_UTILS_OBJ) -o $@

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)
