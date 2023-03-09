#include <stdlib.h>
#include <stdio.h>

#ifndef LEN
#define LEN 15
#endif

#ifndef N
#define N 5
#endif

#ifndef M
#define M 5
#endif

#ifndef PRINT_RESULT_VECTOR
#define PRINT_RESULT_VECTOR(V, NAME) { }
#endif

#ifndef PRINT_RESULT_MATRIX
#define PRINT_RESULT_MATRIX(MAT, NAME) { }
#endif

#define SOLUTION_STACKVEC_1                     \
int a[LEN], b[LEN], c[LEN];                     \
    gettimeofday(&temp_1, (struct timezone*)0); \
    for (int i=0; i<LEN; i++) {                 \
        a[i] = i;                               \
        b[i] = i * 100;                         \
    }                                           \
                                                \
    for (int i=0; i<LEN; i++)                   \
        c[i] = a[i] + b[i];                     \
    gettimeofday(&temp_2, (struct timezone*)0); \
    CPU_times[0] = ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0);


void alloced_heap_vec (int* a, int* b, int* c, int len) {
    for (int i=0; i<len; i++)
        c[i] = a[i] + b[i];

    PRINT_RESULT_VECTOR(c, "c")
}

#define SOLUTION_HEAPVEC_1                      \
    gettimeofday(&temp_1, (struct timezone*)0); \
    int *c1 = (int*)malloc(sizeof(int)*LEN);    \
    alloced_heap_vec(a, b, c1, LEN);            \
    gettimeofday(&temp_2, (struct timezone*)0); \
    CPU_times[1] = ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0);

void not_alloced_heap_vec (int* a, int* b, int** c, int len) {
    (*c) = (int*)malloc(sizeof(int)*len);
    for (int i=0; i<len; i++)
        (*c)[i] = a[i] + b[i];

    PRINT_RESULT_VECTOR((*c), "c")
}

#define SOLUTION_HEAPVEC_2                      \
    gettimeofday(&temp_1, (struct timezone*)0); \
    int *c2;                                    \
    not_alloced_heap_vec(a, b, &c2, LEN);       \
    gettimeofday(&temp_2, (struct timezone*)0); \
    CPU_times[2] = ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0);

int* not_alloced_stack_vec (int* a, int* b, int len) {
    int c[len];
    for (int i=0; i<len; i++)
        c[i] = a[i] + b[i];

    PRINT_RESULT_VECTOR(c, "c")
    return(c);
}

#define SOLUTION_STACKVEC_2      \
    printf("\nNOTE! It isn't possible to compute c as a stack vector of an extern function.\nIn fact, in this case you could return to the main function a pointer to a local variable thet will be freed at the exit of the extern function.\nAs test you can compute it with \"int *c3 = not_alloced_stack_vec(a, b, LEN);\" and then try to print c3 with \"PRINT_RESULT_VECTOR(c3, \"c3\")\": the program will take a segmenatation fault.\n\n");


#define SOLUTION_STACKMAT_1                     \
    gettimeofday(&temp_1, (struct timezone*)0); \
    int A[N][M], B[N][M], C[N][M];              \
    for (int i=0; i<N; i++) {                   \
        for (int j=0; j<M; j++) {               \
            A[i][j] = i + j;                    \
            B[i][j] = (i+j) * 100;              \
            C[i][j] = 0;                        \
        }                                       \
    }                                           \
                                                \
    for (int i=0; i<N; i++) {                   \
        for (int j=0; j<M; j++) {               \
            C[i][j] = A[i][j] + B[i][j];        \
        }                                       \
    }                                           \
    gettimeofday(&temp_2, (struct timezone*)0); \
    CPU_times[4] = ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0);

void alloced_heap_mat (int* A, int* B, int n, int m, int* C) {
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            C[i*m + j] = A[i*m + j] + B[i*m + j];

    PRINT_RESULT_MATRIX(C, "C")
}

#define SOLUTION_HEAPMAT_1                                \
    gettimeofday(&temp_1, (struct timezone*)0);           \
    int *C1 = (int*)malloc(sizeof(int) * N * M);          \
    alloced_heap_mat ((int*)A, (int*)B, N, M, C1);        \
    gettimeofday(&temp_2, (struct timezone*)0);           \
    CPU_times[5] = ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0);

void not_alloced_heap_mat (int* A, int* B, int n, int m, int** C) {
    (*C) = (int*)malloc(sizeof(int)*N*M);
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            (*C)[i*m + j] = A[i*m + j] + B[i*m + j];

    PRINT_RESULT_MATRIX((*C), "C")
}

#define SOLUTION_HEAPMAT_2                              \
    gettimeofday(&temp_1, (struct timezone*)0);         \
    int *C2;                                            \
    not_alloced_heap_mat ((int*)A, (int*)B, N, M, &C2); \
    gettimeofday(&temp_2, (struct timezone*)0);         \
    CPU_times[6] = ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0);

#define SOLUTION_STACKMAT_2   \
    printf("Of course, as for the vector case, it is not possible to compute C as a stack matrix of an extern function.\n\n");

int factorial(int n) {
    int r = 0;

    if (n>1)
        r = n * factorial(n-1);
    else
        r = 1;

    return(r);
}

void heap_factorial (int n, int* r) {
    if (n>1)
        heap_factorial(n-1, r);
    else
        *r = 1;

    (*r) *= n;
    return;
}

// ====================== EX 2 ======================
#include <string.h>


#define EX2_SOLUTION {                          \
    int i;                                      \
    len = 1;                                    \
    for(i=0; i<n; i++) len *= 2;                \
    printf("Vectors len in %d\n", len);         \
    a = (dtype*)malloc(sizeof(dtype)*len);      \
    b = (dtype*)malloc(sizeof(dtype)*len);      \
    c = (dtype*)malloc(sizeof(dtype)*len);      \
    time_t t;                                   \
    srand((unsigned) time(&t));                 \
    int typ = (strcmp( XSTR(dtype) ,"int")==0); \
    if (typ) {                                  \
        int rand_range = (1<<11);               \
        printf("rand_range= %d\n", rand_range); \
        for (i=0; i<len; i++) {                 \
            a[i] = rand()/(rand_range);         \
            b[i] = rand()/(rand_range);         \
            c[i] = a[i] + b[i];                 \
        }                                       \
    } else {                                    \
        for (i=0; i<len; i++) {                 \
        a[i] = (dtype)rand()/((dtype)RAND_MAX); \
        b[i] = (dtype)rand()/((dtype)RAND_MAX); \
        c[i] = a[i] + b[i];                     \
        }                                       \
    }                                           \
}

// ====================== EX 3 ======================
#include "../Laboratory_I/ex3/include/lab1_ex3_lib.h"

vtype* densification_sol (CSR* A) {
    if (A == NULL)
        return(NULL);

    vtype* dn_A = (vtype*)malloc(sizeof(vtype)*(A->n)*(A->m));

    int csr_j;
    for (int i=0; i<A->n; i++) {
        csr_j = A->row[i];
        for (int j=0; j<A->m; j++) {
            if (csr_j < A->row[i + 1] && j==A->col[csr_j]) {
                dn_A[i*A->m + j] = A->val[csr_j];
                csr_j ++;
            } else {
                dn_A[i*A->m + j] = (vtype)0;
            }

        }
    }

    return(dn_A);
}

CSR* sparsification_sol (vtype* A, int n, int m) {
    if (A == NULL)
        return(NULL);

    CSR* csr_A = (CSR*)malloc(sizeof(CSR));
    csr_A->n = n; csr_A->m = m;
    csr_A->row = (itype*)malloc(sizeof(itype)*(n+1));
    csr_A->row[0] = 0;

    int i, j, nnz = 0, row_nnz;
    for (i=0; i<n; i++) {
        row_nnz = 0;
        for (j=0; j<m; j++) {
            if (A[i*m + j] != 0)
                row_nnz++;
        }
        csr_A->row[i+1] = csr_A->row[i] + row_nnz;
        nnz += row_nnz;
    }
    csr_A->nnz = nnz;
    csr_A->col = (itype*)malloc(sizeof(itype)*(nnz));
    csr_A->val = (vtype*)malloc(sizeof(vtype)*(nnz));

    int csr_j = 0;
    for (i=0; i<n; i++) {
        for (j=0; j<m; j++) {
            if (A[i*m + j] != 0) {
                csr_A->col[csr_j] = j;
                csr_A->val[csr_j] = A[i*m + j];
                csr_j++;
            }

            if (csr_j > nnz) {
                printf("ERROR in %s line %d\n", __FILE__, __LINE__);
                exit(1);
            }
        }
    }

    if (csr_j < nnz) {
        printf("ERROR in %s line %d\n", __FILE__, __LINE__);
        exit(1);
    }

    return(csr_A);
}

#define EX3_SOLUTION {                                  \
    dn_A = densification_sol(A);                 \
    ck_A = sparsification_sol(dn_A, A->n, A->m);   \
}
