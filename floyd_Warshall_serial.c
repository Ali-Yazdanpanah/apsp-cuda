
#include "floyd_Warshall_serial.h"

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
