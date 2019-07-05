#include <stdio.h>
#include <time.h>
#include <math.h>

#define MAX_W 9999999
#define TRUE    1
#define FALSE   0
typedef int boolean;

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


Vertex *vertices;	
Edge *edges;

int findEdge(Vertex u, Vertex v, Edge *edges, int *weights, int E)
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



void Graph_Randomizer(int V, int E){
	srand(time(NULL));
	
	for(int i = 0; i < V; i++)
	{
		Vertex a = { .title =(int) i, .visited=FALSE};
		vertices[i] = a;
	}
	for(i = 0; i < E; i++)
	{
		Edge e = {.u = (int) rand()%V , .v = rand()%V};
		edges[i] = e;
		weights[i] = rand()%100;
	}
}


int main(int argc, char **argv)
{
	V = atoi(argv[1]);
	E = atoi(argv[2]);

	int *weights;
	int *len, *updateLength;
	Vertex *d_V;
	Vertex *root;
	Edge *d_E;
	int *d_W;
	int *d_L;
	int *d_C;
	float runningTime;
	vertices = (Vertex *)malloc(sizeof(Vertex) * V);
	edges = (Edge *)malloc(sizeof(Edge) * E);
	Graph_Randomizer(V, E);
	weights = (int *)malloc(E* sizeof(int));
	len = (int *)malloc(V * sizeof(int));
	updateLength = (int *)malloc(V * sizeof(int));
	root = (Vertex *)malloc(sizeof(Vertex) * V);
	for(int count = 0; count < V; count++)
		root[count] = {count,FALSE};
    clock_t start = clock();
	for(int count = 0; count < V; count++){
		root[count].visited = TRUE;
		len[root[count].title] = 0;
		updateLength[root[count].title] = 0;
		for(i = 0; i < V;i++)
		{
			if(vertices[i].title != root[count].title)
			{
				len[(int)vertices[i].title] = findEdge(root, vertices[i], edges, weights, E);
				updateLength[vertices[i].title] = len[(int)vertices[i].title];
			}
			else{
				vertices[i].visited = TRUE;
			}
		}	
		for(int i = 0; i < V; i++){
                if(vertices[i].visited == FALSE)
                    {
                        vertices[i].visited = TRUE;
                        for(int v = 0; v < V; v++)
                        {				
                            int weight = findEdge(vertices[i], vertices[v], edges, weights);
                            if(weight < MAX_W)
                            {	
                                if(updateLength[v] > length[i] + weight)
                                {
                                    updateLength[v] = length[i] + weight;
                                }
                            }
                        }

                    }
				for(int j = 0; j < V; j++)
				{
                    if(length[j] > updateLength[j])
                        {
                            vertices[j].visited = FALSE;
                            length[j] = updateLength[j];
                        }
                        updateLength[j] = length[j];
                    }
		}	
	}
    	
	clock_t end = clock();   
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time on CPU = %f sec\n", seconds);
}



	
	
	
	
	
	
	
	
	
	
	
	
	

