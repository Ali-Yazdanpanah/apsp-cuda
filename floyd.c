
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INFTY 99999

int **A_Matrix, **D_Matrix;

void makeAdjacency(int n){   //Set initial values to node distances
    int N=n;
    int i,j;
    A_Matrix = (int **)malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        A_Matrix[i] = (int *) malloc(N * sizeof(int));
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
void floyd_warshall_serial (int** graph, int** dist, int N) {
	int i, j, k;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			dist[i][j] = graph[i][j];
	for (k = 0; k < N; k++) {
        	for (i = 0; i < N; i++) {
            		for (j = 0; j < N; j++) {
                		if (dist[i][k] + dist[k][j] < dist[i][j])
                    			dist[i][j] = dist[i][k] + dist[k][j];
            		}
        	}
    	}
}


int main(int argc, char **argv){
    int N;     
	N = atoi(argv[1]);  //Read the console inputs
    int i, j, k;
	D_Matrix =(int **) malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        D_Matrix[i] =(int *) malloc(N * sizeof(int));
    }
    makeAdjacency(N);
    clock_t start = clock();  //First time measurement
    // algorithm -->
    floyd_warshall_serial(A_Matrix,D_Matrix,N);
    // <--;	
	clock_t end = clock();   //Final time calculation and convert it into seconds
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed time = %f sec\n", seconds);
	
}



