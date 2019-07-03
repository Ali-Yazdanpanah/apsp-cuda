
#include "Dijkstra.h" 
   
int **A_Matrix, **D_Matrix;
bool **stpSet;

// A utility function to find the vertex with minimum distance value, from 
// the set of vertices not yet included in shortest path tree 
int minDistance(int* dist, bool* stpSet, int n) 
{ 
   // Initialize min value 
    int min = INT_MAX, min_index; 
    for(int j = 0; j < n; j++){
        if (stpSet[j] == false && dist[j] <= min) 
        min = dist[j], min_index = j;
    } 
    return min_index; 
} 


void makeAdjacency(int n){   //Set initial values to node distances
    int N=n;
    int i,j;
    A_Matrix = (int **)malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        A_Matrix[i] = (int *)malloc(N * sizeof(int));
    }
    srand(0);
    for(i = 0; i < N; i++)
    {
        for(j = i; j < N; j++)
        {
            if(i == j){
                A_Matrix[i][j] = 0;
            }
            else{
                int r = rand() % 10;
                int val = (r == 5)? INFTY: r;
                A_Matrix[i][j] = val;
                A_Matrix[j][i] = val; 
            }

        }
    }
}




void dijkstra_all(int** graph, int** dist, bool** stpSet, int n){
    for(int i = 0: i < n; i++){
        dijkstra(graph, dist[i],stpSet[i],n,i);
    }
}

// Function that implements Dijkstra's single source shortest path algorithm 
// for a graph represented using adjacency matrix representation 
void dijkstra(int** graph,int* dist,bool* stpSet, int n, int src) 
{      // The output array.  dist[i] will hold the shortest 
                      // distance from src to i 
   
     bool stpSet[n]; // stpSet[i] will be true if vertex i is included in shortest 
                     // path tree or shortest distance from src to i is finalized 
   
     // Initialize all distances as INFINITE and stpSet[] as false 
     for (int i = 0; i < n; i++) 
        dist[i] = INT_MAX, stpSet[i] = false;  
     dist[src] = 0; 
     for (int count = 0; count < n-1; count++) 
     { 
       // Pick the minimum distance vertex from the set of vertices not 
       // yet processed. u is always equal to src in the first iteration. 
       int u = minDistance(dist, stpSet); 
   
       // Mark the picked vertex as processed 
       stpSet[u] = true; 
   
       // Update dist value of the adjacent vertices of the picked vertex. 
       for (int v = 0; v < n; v++) 
   
         // Update dist[v] only if is not in stpSet, there is an edge from  
         // u to v, and total weight of path from src to  v through u is  
         // smaller than current value of dist[v] 
         if (!stpSet[v] && graph[u][v] && dist[u] != INT_MAX  
                                       && dist[u]+graph[u][v] < dist[v]) 
            dist[v] = dist[u] + graph[u][v]; 
     } 
} 
   
// driver program to test above function 
int main(int argc, char **argv) 
{ 
    int N;     
	N = atoi(argv[1]);  //Read the console inputs
    int i, j, k;
	D_Matrix =(int **) malloc(N * sizeof(int *));
    stpSet = (bool **) malloc(N * sizeof(bool *));
    for (i = 0; i < N; i++)
    {
        D_Matrix[i] = (int *)malloc(N * sizeof(int));
        stpSet[i] = (bool *)malloc(N * sizeof(bool))
    }
    makeAdjacency(N);
    clock_t start = clock();  //First time measurement
    // algorithm -->
    dijkstra_all(A_Matrix,D_Matrix,stpSet,N);
    // <--;	
	clock_t end = clock();   //Final time calculation and convert it into seconds
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time = %f sec\n", seconds);
    return 0; 
} 