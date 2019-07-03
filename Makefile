nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -D TEST --compiler-options -Wall 
sources1 = main.c floyd_Warshall_serial.c
sources2 = Dijkstra.c 
all: apsp1 apsp2 apsp1cuda



apsp1:
	gcc -std=c99 $(sources1) -o apsp1 -lm

apsp2:
	gcc -std=c99 $(sources2) -o apsp2 -lm

apsp1cuda:
	nvcc main.cu -o apsp1cuda $(nvcc_options)

clean:
	rm -rf apsp1 apsp2 apsp1cuda
