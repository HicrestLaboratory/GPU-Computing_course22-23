#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define DBG_CHKP { printf("DBG_CHKP: %s %d\n", __func__, __LINE__); }

#include "include/lab1_ex3_lib.h"
// NOTE: vtype, itype and CSR struct are defined in the header file 'include/lab1_ex3_lib.h'

// -------- uncomment these two lines when solutions are published --------
// #include "../../solutions/lab1_sol.cu"
// #define RESULTS
// ------------------------------------------------------------------------

CSR* random_CSR (int n, int m) {
    time_t t;
    srand((unsigned) time(&t));

#ifdef DEBUG
    printf("============= random generation printing =============\n");
#endif

    CSR* A = (CSR*)malloc(sizeof(CSR));
    A->n = n; A->m = m;
    A->row = (itype*)malloc(sizeof(itype)*(n+1));
    A->row[0] = 0;

    int row_nnz, nnz = 0, rand_extraction;
    for (int i=0; i<n; i++) {

        row_nnz = 1;
        rand_extraction = rand();
        while ((RAND_MAX/(1<<row_nnz) > rand_extraction) && (row_nnz<m-1) )
            row_nnz ++;
#ifdef DEBUG
        printf("row_nnz[%d] = %d\n", i, row_nnz);
#endif

        A->row[i+1] = A->row[i] + row_nnz;
        nnz += row_nnz;
    }
    A->nnz = nnz;

    A->col = (itype*)malloc(sizeof(itype)*(nnz));
    A->val = (vtype*)malloc(sizeof(vtype)*(nnz));
    int tmp, k;
    for (int i=0; i<n; i++) {
#ifdef DEBUG
        printf("----------------- row %2d generation ------------------\n", i);
#endif

        for (int j=0; j<(A->row[i+1]-A->row[i]); j++) {
            A->val[A->row[i] + j] = (double)rand()/((double)(RAND_MAX/5));
            rand_extraction = (m-j!=0) ? rand() % (m-j) : 0 ;

            for (k=0; k<j; k++) {
                if (A->col[A->row[i] + k] <= rand_extraction) {
                    rand_extraction++;
                } else {
                    break;
                }
            }

#ifdef DEBUG
            printf("rand_extraction = %d\n", rand_extraction);
#endif

            if (k<j) {

#ifdef DEBUG
                printf("k = %d, j = %d\n", k, j);
                for (int z=A->row[i]; z<A->row[i+1]; z++)
                    printf("%d ", A->col[z]);
                printf("\n");
#endif

                for (int h=j; h>k; h--) {
                    A->col[A->row[i] + h] = A->col[A->row[i] + h -1];
                }

#ifdef DEBUG
                for (int z=A->row[i]; z<A->row[i+1]; z++)
                    printf("%d ", A->col[z]);
                printf("\n");
#endif
            }

            A->col[A->row[i] + k] = rand_extraction;

#ifdef DEBUG
            printf("A[%d]: ", i);
            for (int z=A->row[i]; z<A->row[i+1]; z++)
                printf("%d ", A->col[z]);
            printf("\n");
#endif
        }
    }

#ifdef DEBUG
    printf("------------------------------------------------------\n");
    printf("======================================================\n\n");
#endif

    return(A);
}

int compare_CSR (CSR* A, CSR* B) {
    if (A == NULL || B == NULL)
        return(0);

    int result = 1;
    if ((A->n == B->n) && (A->m == B->m) && (A->nnz == B->nnz)) {
        if (result) {
            if (memcmp(A->row, B->row, sizeof(itype) * (A->n +1)) != 0)
                result = 0;
            if (result) {
                if (memcmp(A->col, B->col, sizeof(itype) * (A->nnz)) != 0)
                    result = 0;
                if (result)
                    if (memcmp(A->val, B->val, sizeof(vtype) * (A->nnz)) != 0)
                        result = 0;
            }
        }
    } else {
        result = 0;
    }

    return(result);
}



void free_CSR (CSR** A) {
    if ((*A) != NULL) {
        free((*A)->row);
        free((*A)->col);
        free((*A)->val);
        free((*A));
        (*A) = NULL;
    }
}

void print_CSR (CSR* A, const char* Aname) {
    printf("Sparse matrix %s:\n", Aname);
    printf("A->n = %d\n", A->n);
    printf("A->m = %d\n", A->m);
    printf("A->nnz = %d\n", A->nnz);
    printf("Rows: ");
    for (int i=0; i<(A->n+1); i++)
        printf("%3d ", A->row[i]);
    printf("\n");
    printf("Cols: ");
    for (int i=0; i<A->nnz; i++)
        printf("%3d ", A->col[i]);
    printf("\n");
    printf("Vals: ");
    for (int i=0; i<A->nnz; i++)
        printf("%3.2f ", A->val[i]);
    printf("\n\n");
}

void print_DN (vtype* A, int n, int m) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++)
            printf("%3.2f ", A[i*m + j]);
        printf("\n");
    }
    printf("\n");

    return;
}

int main (int argc, char *argv[]) {

    if (argc < 3) {
        printf("Usage: lab1_ex2 n m\n");
        return(1);
    }

    printf("argv[1] = %s, argv[2] = %s\n", argv[1], argv[2]);

    int n = atoi(argv[1]), m = atoi(argv[2]);

    CSR* A = random_CSR(n,m);

    print_CSR(A, "A");

    vtype* dn_A = NULL;  // Put here the result of your densification function over A
    CSR* ck_A = NULL;    // Put here the result of your sparsification function over dn_A

#ifdef RESULTS
    EX3_SOLUTION
#else
        /* |========================================| */
        /* |           Put here your code           | */
        /* |========================================| */



#endif

    int test = compare_CSR(A, ck_A);
    char* test_result = (test == 1) ? "\x1B[32mDONE!\x1B[37m" : "\x1B[31mERROR!\x1B[37m";
    printf("\nINVERSE TEST (|sparsification(densification(A)) == A ?): \t %s\n", test_result);

    free_CSR(&ck_A);
    free_CSR(&A);
    if (dn_A != NULL)
        free(dn_A);

    return(0);

}
