nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -D TEST --compiler-options -Wall 

all: apsp1 apsp2 apsp1cuda apsp2cuda



apsp1:
	gcc -std=c99 floyd.c -o apsp1 -lm

apsp2:
	gcc -std=c99 Dijkstra.c -o apsp2 -lm

apsp1cuda:
	nvcc floyd.cu -o apsp1cuda $(nvcc_options)

apsp2cuda:
	nvcc Dijkstra.cu -o apsp2cuda $(nvcc_options)

clean:
	rm -rf apsp1 apsp2 apsp1cuda apsp2cuda
