# Laboratory I: exercise3

Given the random CSR matrix ‘M’ generated in “ex3/lab1_ex3.c”, implement the functions:
1. “densification()” which take in input a CSR matrix A and return in output his dense version dn_A.
2. “sparsification()” which take in input a dense matrix A and return in output his CSR version csr_A.

After this, check that “sparsification(densification(M)) == M” by using “int compare_CSR (CSR* A, CSR* B)”.
