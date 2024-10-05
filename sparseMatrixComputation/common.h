// different fields in COOmatrix

struct COOMatrix
{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonZeros;
    unsigned int* rowIdxs;
    unsigned int* colIdxs;
    float* values;
};


struct CSRMatrix
{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonZeros;
    unsigned int* rowPtrs;
    unsigned int* colIdxs;
    float* values;
};

struct ELLMatrix
{
    unsigned int numRows;
    unsigned int numCols;
    unsigned int maxNNZPerRow;
    unsigned int* nnzPerRow;
    unsigned int* colIdxs;
    float* values;
};


