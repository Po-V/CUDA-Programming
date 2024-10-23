struct CSRGraph{
    unsigned int numVertices;
    unsigned int numEdges;
    unsigned int* srcPtrs;
    unsigned int* dst;
};

struct COOGraph{
    unsigned int numVertices;
    unsigned int numEdges;
    unsigned int* src;
    unsigned int* dst;
};