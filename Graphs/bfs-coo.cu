#include "common.h"

// use csr representation for vertex centric approach
__global__ void bfs_kernel(COOGraph cooGraph, unsigned int* level, unsigned int* newVertexVisited, unsigned int currLevel){
    // determine which vertex a thread is responsible for
    unsigned int edge = blockIdx.x*blockDim.x+threadIdx.x;
    // check edge is within bound
    if(edge < cooGraph.numEdges){
        
        unsigned int src = cooGraph.src[edge];
        unsigned int dst = cooGraph.dst[edge];

        if(level[src] == currLevel - 1 && level[dst] == UINT_MAX){
            level[dst] =  currLevel;
            *newVertexVisited = 1;
        }
    }
}

void bfs_gpu(COOGraph cooGraph, unsigned int srcVertex, unsigned int* level){

    // allocate GPU memory
    COOGraph cooGraph_d;
    cooGraph_d.numVertices = cooGraph.numVertices;
    cooGraph_d.numEdges = cooGraph.numEdges;
    cudaMalloc((void**) &cooGraph_d.src, cooGraph_d.numEdges*sizeof(unsigned int));
    cudaMalloc((void**) &cooGraph_d.dst, cooGraph_d.numEdges*sizeof(unsigned int));
    unsigned int* level_d;
    cudaMalloc((void**) &level_d, cooGraph_d.numVertices*sizeof(unsigned int));
    unsigned int* newVertexVisited_d;
    cudaMalloc((void**) &newVertexVisited_d, sizeof(unsigned int));
    cudaDeviceSynchronize();
    
    // copy data to GPU
    cudaMemcpy(cooGraph_d.src, cooGraph.src, cooGraph_d.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cooGraph_d.dst, cooGraph.dst, cooGraph_d.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    level[srcVertex] = 0;
    cudaMemcpy(level_d, level, cooGraph_d.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // call kernel
    unsigned int numThreadsPerBlock = 128;
    unsigned int numBLocks = (cooGraph_d.numEdges + numThreadsPerBlock - 1)/numThreadsPerBlock;
    unsigned int newVertexVisited = 1;
    for(unsigned int currLevel = 1; newVertexVisited; ++currLevel){
        newVertexVisited = 0;
        cudaMemcpy(newVertexVisited_d, &newVertexVisited, sizeof(unsigned int), cudaMemcpyHostToDevice);
        bfs_kernel<<<numBLocks, numThreadsPerBlock>>>(cooGraph_d, level_d, newVertexVisited_d, currLevel);
        cudaMemcpy(&newVertexVisited, newVertexVisited_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(level, level_d, cooGraph_d.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free GPU memory
    cudaFree(cooGraph_d.src);
    cudaFree(cooGraph_d.dst);
    cudaFree(level_d);
    cudaFree(newVertexVisited_d);
}

int main(){

    // example graph in CSR format
    COOGraph graph;
    graph.numVertices = 4;
    graph.numEdges = 8;

    int adjacencyMatrix[graph.numVertices][graph.numVertices] = {
        {0, 1, 1, 0},
        {1, 0, 1, 1},
        {1, 1, 0, 0},
        {0, 1, 0, 0}
    };


    graph.src = (unsigned int* )malloc(graph.numEdges*sizeof(unsigned int));
    graph.dst = (unsigned int*)malloc(graph.numEdges*sizeof(unsigned int));

    // Fill in the COO data
    int edgeIndex = 0;
    for (int i = 0; i < graph.numVertices; i++) {
        for (int j = 0; j < graph.numVertices; j++) {
            if (adjacencyMatrix[i][j] == 1) {
                graph.src[edgeIndex] = i;
                graph.dst[edgeIndex] = j;
                edgeIndex++;
            }
        }
    }

    // Allocate and initialize level array. Level is distance from source vertex
    unsigned int* level = (unsigned int*)malloc(graph.numVertices * sizeof(unsigned int));
    for (unsigned int i = 0; i < graph.numVertices; i++) {
        level[i] = UINT_MAX;
    }

    // choose a source vertex
    unsigned int srcVertex = 0;

    // run BFS on GPU
    bfs_gpu(graph, srcVertex, level);

    free(graph.src);
    free(graph.dst);
    free(level);

    return 0;
}



