# PrefixSum and SpMM

Exercise 1: prefix sum of the even elements

1. Take n as input and generate an arbitrary input vector of length 2n
2. Implement a cuda kernel that perform the prefix-sum only over the even element of the input vector
3. Measure and print the running time and the bandwidth of your kernel


Exercise 2: sparse matrix product

1. Take (n, m, nnz) as inputs and generate a random CSR A of dimension n x m with nnz non-zero value
2. Generate the dense matrix B of dimension m x m such that each element of B is 1
3. Write a kernel that compute the product C = A ⋅ B
4. Check the correctness of C by verifying that each element is the sum of the corresponding row of A
5. Measure the kernel Flops and Bandwidth and compare them with the GPU theoretical peak

