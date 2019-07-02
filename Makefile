nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -D TEST --compiler-options -Wall 
sources = main.cu floyd_Warshall_serial.c

all: apsp1 apsp1cuda

apsp1cuda:
	$(sources) Makefile
	nvcc -o apsp1cuda $(sources) $(nvcc_options)

clean:
	rm apsp1cuda
