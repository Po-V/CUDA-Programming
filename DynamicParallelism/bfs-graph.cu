#include "common.h"

// process iterations in parallel
__global__ void bfs_child_kernel(CSRGraph csrGraph, unsigned int* level, unsigned int* currFrontier, unsigned int numPrevFrontier, 
unsigned int* numCurrFrontier, unsigned int currLevel, unsigned int numNeighbors, unsigned int start){
    
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    // every thread takes one neighbor
    if(i < numNeighbors){
        unsigned int edge = start + i;
        unsigned int neighbor = csrGraph.dst[edge];
        if(atomicCAS(&level[neighbor], UINT_MAX, currLevel) == UINT_MAX){
            unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
            currFrontier[currFrontierIdx] = neighbor;
        }
    }
}

// use csr representation for vertex centric approach
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level, unsigned int* prevFrontier, unsigned int* currFrontier, unsigned int numPrevFrontier, unsigned int* numCurrFrontier, unsigned int currLevel){
    
    // used threads to index previous frontier
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    // boundaary check with number of previous frontier
    if( i < numPrevFrontier){
        // threads access vertex it is responsible for
        unsigned int vertex = prevFrontier[i];
        unsigned int start = csrGraph.srcPtrs[vertex];
        unsigned int numNeighbors = csrGraph.srcPtrs[vertex + 1] - start;
        unsigned int numThreadsPerBlock = 1024;
        // assign threads in threadblock to neighbors
        unsigned int numBlocks = (numNeighbors + numThreadsPerBlock - 1)/numThreadsPerBlock;
        bfs_child_kernel<<<numBlocks, numThreadsPerBlock>>>(csrGraph, level, currFrontier, numPrevFrontier, numCurrFrontier, currLevel, numNeighbors, start);
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
    unsigned int* buffer1_d;
    unsigned int* buffer2_d;
    cudaMalloc((void**) &buffer1_d, csrGraph_d.numVertices*sizeof(unsigned int));
    cudaMalloc((void**) &buffer2_d, csrGraph_d.numVertices*sizeof(unsigned int));
    unsigned int* numCurrFrontier_d;
    cudaMalloc((void**) &numCurrFrontier_d, sizeof(unsigned int));
    unsigned int* prevFrontier_d = buffer1_d;
    unsigned int* currFrontier_d = buffer2_d;
    cudaDeviceSynchronize();
    
    // copy data to GPU
    cudaMemcpy(csrGraph_d.srcPtrs, csrGraph.srcPtrs, (csrGraph_d.numVertices +1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrGraph_d.dst, csrGraph.dst, csrGraph_d.numEdges*sizeof(unsigned int), cudaMemcpyHostToDevice);
    level[srcVertex] = 0;
    cudaMemcpy(level_d, level, csrGraph_d.numVertices*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(prevFrontier_d, &srcVertex, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // call kernel
    unsigned int numPrevFrontier = 1;
    unsigned int numThreadsPerBlock = 256;
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, csrGraph.numVertices);
    for(unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel){
        
        // visit vertices in previous frontier
        cudaMemset(numCurrFrontier_d, 0, sizeof(unsigned int));
        unsigned int numBLocks = (numPrevFrontier + numThreadsPerBlock - 1)/numThreadsPerBlock;
        bfs_kernel<<<numBLocks, numThreadsPerBlock>>>(csrGraph_d, level_d, prevFrontier_d, currFrontier_d, numPrevFrontier, numCurrFrontier_d, currLevel);

        // swap buffers
        unsigned int* tmp = prevFrontier_d;
        prevFrontier_d = currFrontier_d;
        currFrontier_d = tmp;
        cudaMemcpy(&numPrevFrontier, numCurrFrontier_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    // copy data from GPU
    cudaMemcpy(level, level_d, csrGraph_d.numVertices*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // free GPU memory
    cudaFree(csrGraph_d.srcPtrs);
    cudaFree(csrGraph_d.dst);
    cudaFree(level_d);
    cudaFree(buffer1_d);
    cudaFree(buffer2_d);
    cudaFree(numCurrFrontier_d);
    cudaDeviceSynchronize();
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
