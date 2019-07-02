nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -D TEST --compiler-options -Wall 
sources = main.c floyd_warshall_serial.c

all: apsp1

apsp1: $(sources) Makefile floyd_warshall_serial.h
	gcc -std=c99 -o apsp1 $(sources)

clean:
	rm apsp1