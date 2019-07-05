#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_W 9999999
#define TRUE    1
#define FALSE   0
typedef int boolean;
int E,V;
typedef struct
{
	int u;
	int v;
} Edge;

typedef struct 
{
	int title;
	boolean visited;	
} Vertex;

int *weights;
Vertex *vertices;	
Edge *edges;
//Finds the weight of the path from vertex u to vertex v
__device__ __host__ int findEdge(Vertex u, Vertex v, Edge *edges, int *weights, int E)
{
	for(int i = 0; i < E; i++)
	{
		if(edges[i].u == u.title && edges[i].v == v.title)
		{
			return weights[i];
		}
	}
	return MAX_W;
}


__global__ void Find_Vertex(Vertex *vertices, Edge *edges, int *weights, int *length, int *updateLength, int V, int E)
{
	int u = threadIdx.x;
	if(vertices[u].visited == FALSE)
	{
		vertices[u].visited = TRUE;
		for(int v = 0; v < V; v++)
		{				
			int weight = findEdge(vertices[u], vertices[v], edges, weights, E);
			if(weight < MAX_W)
			{	
				if(updateLength[v] > length[u] + weight)
				{
					updateLength[v] = length[u] + weight;
				}
			}
		}

	}
	
}

//Updates the shortest path array (length)
__global__ void Update_Paths(Vertex *vertices, int *length, int *updateLength)
{
	int u = threadIdx.x;
	if(length[u] > updateLength[u])
	{
		vertices[u].visited = FALSE;
		length[u] = updateLength[u];
	}
	updateLength[u] = length[u];
}



void Graph_Randomizer(int V, int E){
	srand(time(NULL));
	
	for(int i = 0; i < V; i++)
	{
		Vertex a = { .title =(int) i, .visited=FALSE};
		vertices[i] = a;
	}
	for(int i = 0; i < E; i++)
	{
		Edge e = {.u = (int) rand()%V , .v = rand()%V};
		edges[i] = e;
		weights[i] = rand()%100;
	}
}

//Runs the program
int main(int argc, char **argv)
{
	V = atoi(argv[1]);
	E = atoi(argv[2]);


	int *len, *updateLength;
	Vertex *d_V;
	Vertex *root;
	Edge *d_E;
	int *d_W;
	int *d_L;
	int *d_C;
	vertices = (Vertex *)malloc(sizeof(Vertex) * V);
    edges = (Edge *)malloc(sizeof(Edge) * E);
    weights = (int *)malloc(E* sizeof(int));
	Graph_Randomizer(V, E);
	len = (int *)malloc(V * sizeof(int));
	updateLength = (int *)malloc(V * sizeof(int));
	root = (Vertex *)malloc(sizeof(Vertex) * V);
	cudaMalloc((void**)&d_V, sizeof(Vertex) * V);
	cudaMalloc((void**)&d_E, sizeof(Edge) * E);
	cudaMalloc((void**)&d_W, E * sizeof(int));
	cudaMalloc((void**)&d_L, V * sizeof(int));
    cudaMalloc((void**)&d_C, V * sizeof(int));
    cudaMemcpy(d_V, vertices, sizeof(Vertex) * V, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E, edges, sizeof(Edge) * E, cudaMemcpyHostToDevice);
	cudaMemcpy(d_W, weights, E * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, len, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, updateLength, V * sizeof(int), cudaMemcpyHostToDevice);
	for(int count = 0; count < V; count++){
        root[count].title = count;
        root[count].visited = FALSE;   
    }
    clock_t start = clock();
	for(int count = 0; count < V; count++){
		root[count].visited = TRUE;
		len[root[count].title] = 0;
		updateLength[root[count].title] = 0;
		for(int i = 0; i < V;i++)
		{
			if(vertices[i].title != root[count].title)
			{
                len[(int)vertices[i].title] = findEdge(root[count], vertices[i], edges, weights, E);
				updateLength[vertices[i].title] = len[(int)vertices[i].title];
			}
			else{
				vertices[i].visited = TRUE;
			}
		}	
		cudaMemcpy(d_L, len, V * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_C, updateLength, V * sizeof(int), cudaMemcpyHostToDevice);
		for(int i = 0; i < V; i++){
				Find_Vertex<<<1, V>>>(d_V, d_E, d_W, d_L, d_C, V, E);
				for(int j = 0; j < V; j++)
				{
					Update_Paths<<<1,V>>>(d_V, d_L, d_C);
				}
		}	
    }
	clock_t end = clock();   
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time on GPU = %f sec\n", seconds);
}



	
	
	
	
	
	
	
	
	
	
	
	
	

