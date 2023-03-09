#ifndef LAB1_EX3_LIB
#define LAB1_EX3_LIB

#define itype int
#define vtype double

typedef struct CSR {
    int n;
    int m;
    int nnz;
    itype* col;
    vtype* val;
    itype* row;
} CSR;

#endif
