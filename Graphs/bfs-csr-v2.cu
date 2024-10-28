#include "common.h"

// use csr representation for vertex centric approach
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level, unsigned int* newVertexVisited, unsigned int currLevel){
    // determine which vertex a thread is responsible for
    unsigned int vertex = blockIdx.x*blockDim.x+threadIdx.x;
    if(vertex < csrGraph.numVertices){
        // if vertex have not been visited
        if(level[vertex] == UINT_MAX){
            // loop over nehghbours and check if they are in previous level
            for(unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge){
                // get neighbor
                unsigned int neighbor = csrGraph.dst[edge];
                if(level[neighbor] == currLevel - 1){
                    // mark current vertex as current level
                    level[vertex] = currLevel;
                    // flag to indicate new vertex was visited
                    *newVertexVisited = 1;
                    // the current level vertex was found
                    break;
                }
            }
        }
    }
}

void bfs_gpu(CSRGraph csrGraph, unsigned int srcVertex, unsigned int* level){

    // allocate GPU memory
    CSRGraph csrGraph_d;
    csrGraph_d.numVertices = csrGraph.numVertices;
    csrGraph_d.numEdges = csrGraph.numEdges;
    cudaMalloc((void**) &csrGraph_d.srcPtrs, (csrGraph_d.numVertices +1)*sizeof(unsigned int));
    cudaMalloc((void**) &csrGraph_d.dst, csrGraph_d.numEdges*sizeof(unsigned int));
    unsigned int* level_d;
    cudaMalloc((void**) &level_d, csrGraph_d.numVertices*sizeof(unsigned int));
    unsigned int* newVertexVisited_d;
    cudaMalloc((void**) &newVertexVisited_d, sizeof(unsigned int));
    cudaDeviceSynchronize();
    
    // copy data to GPU
    cudaMemcpy(csrGraph_d.srcPtrs, csrGraph.srcPtrs, (csrGraph_d.numVertices +1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrGraph_d.dst, csrGraph.dst, csrGraph_d.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    level[srcVertex] = 0;
    cudaMemcpy(level_d, level, csrGraph_d.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // call kernel
    unsigned int numThreadsPerBlock = 128;
    unsigned int numBLocks = (csrGraph_d.numVertices + numThreadsPerBlock - 1)/numThreadsPerBlock;
    unsigned newVertexVisited = 1;
    for(unsigned int currLevel = 1; newVertexVisited; ++currLevel){
        newVertexVisited = 0;
        cudaMemcpy(newVertexVisited_d, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
        bfs_kernel<<<numBLocks, numThreadsPerBlock>>>(csrGraph_d, level_d, newVertexVisited_d, currLevel);
        cudaMemcpy(&newVertexVisited, newVertexVisited_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(level, level_d, csrGraph_d.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free GPU memory
    cudaFree(csrGraph_d.srcPtrs);
    cudaFree(csrGraph_d.dst);
    cudaFree(level_d);
}

int main(){

    // example graph in CSR format
    CSRGraph graph;
    graph.numVertices = 4;
    graph.numEdges = 8;

    int adjacencyMatrix[graph.numVertices][graph.numVertices] = {
        {0, 1, 1, 0},
        {1, 0, 1, 1},
        {1, 1, 0, 0},
        {0, 1, 0, 0}
    };


    graph.srcPtrs = (unsigned int* )malloc((graph.numVertices + 1)*sizeof(unsigned int));
    graph.dst = (unsigned int*)malloc(graph.numEdges*sizeof(unsigned int));

    // Fill in the CSR data using loops
    unsigned int edgeCount = 0;
    for (int i = 0; i < graph.numVertices; i++) {
        graph.srcPtrs[i] = edgeCount;
        for (int j = 0; j < graph.numVertices; j++) {
            if (adjacencyMatrix[i][j] == 1) {
                graph.dst[edgeCount] = j;
                edgeCount++;
            }
        }
    }
    graph.srcPtrs[graph.numVertices] = edgeCount;

    // Allocate and initialize level array. Level is distance from source vertex
    unsigned int* level = (unsigned int*)malloc(graph.numVertices * sizeof(unsigned int));
    for (unsigned int i = 0; i < graph.numVertices; i++) {
        level[i] = UINT_MAX;
    }

    // choose a source vertex
    unsigned int srcVertex = 0;

    // run BFS on GPU
    bfs_gpu(graph, srcVertex, level);

    free(graph.srcPtrs);
    free(graph.dst);
    free(level);

    return 0;
}


