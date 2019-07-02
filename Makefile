nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -D TEST --compiler-options -Wall 
sources = main.c floyd_Warshall_serial.c

all: apsp1 apsp1cuda

apsp1:
	gcc -std=c99 $(sources) -o apsp1 -lm

apsp1cuda:
	main.cu Makefile
	nvcc -o main.cu $(nvcc_options)

clean:
	rm apsp1cuda apsp1
