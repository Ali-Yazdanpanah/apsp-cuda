nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -D TEST --compiler-options -Wall 
sources = main.c floyd_Warshall_serial.c

all: apsp1

apsp1: $(sources) floyd_Warshall_serial.h
	gcc -std=c99 -o apsp1 $(sources) -lm

clean:
	rm apsp1
