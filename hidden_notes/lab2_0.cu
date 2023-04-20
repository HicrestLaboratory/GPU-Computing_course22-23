#include <stdio.h>
#include <stdlib.h>

#define MY_CUDA_CHECK( call) {                                    \
	    cudaError err = call;                                                    \
	    if( cudaSuccess != err) {                                                \
		            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
					                    __FILE__, __LINE__, cudaGetErrorString( err) );              \
		            exit(EXIT_FAILURE);                                                  \
		        } }

#define N 5

__global__ void vectorAdd (float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];

}

int main (void) {
    int n = N, i;
    int len = 1 << n;

    float *h_a = (float*) malloc(sizeof(float) * len);
    float *h_b = (float*) malloc(sizeof(float) * len);
    float *h_c = (float*) malloc(sizeof(float) * len);

    for (i=0; i<len; i++) {
        h_a[i] = 2.0;
        h_b[i] = 1.0;
    }

    printf("h_a: ");
    for (i=0; i<len; i++)
        printf("%f ", h_a[i]);
    printf("\n");
    printf("h_b: ");
    for (i=0; i<len; i++)
        printf("%f ", h_b[i]);
    printf("\n");

    float *d_a, *d_b, *d_c;
    MY_CUDA_CHECK( cudaMalloc(&d_a, sizeof(float) * len) );
    MY_CUDA_CHECK( cudaMalloc(&d_b, sizeof(float) * len) );
    MY_CUDA_CHECK( cudaMalloc(&d_c, sizeof(float) * len) );

    MY_CUDA_CHECK( cudaMemcpy(d_a, h_a, sizeof(float) * len, cudaMemcpyHostToDevice) );
    MY_CUDA_CHECK( cudaMemcpy(d_b, h_b, sizeof(float) * len, cudaMemcpyHostToDevice) );
    MY_CUDA_CHECK( cudaMemset(d_c, 0, sizeof(float) * len) );

    int threds_per_block = 256;
    int blocks_per_grid = (len + threds_per_block -1) / threds_per_block;
    printf("len = %d, blocks_per_grid = %d, threds_per_block = %d\n", len, blocks_per_grid, threds_per_block);
    vectorAdd<<<blocks_per_grid, threds_per_block>>>(d_a, d_b, d_c, len);


    MY_CUDA_CHECK(cudaMemcpy(h_c, d_c, sizeof(float) * len, cudaMemcpyDeviceToHost) );

    MY_CUDA_CHECK(cudaFree(d_a));
    MY_CUDA_CHECK(cudaFree(d_b));
    MY_CUDA_CHECK(cudaFree(d_c));

    printf("h_c: ");
    for (i=0; i<len; i++)
        printf("%f ", h_c[i]);
    printf("\n");

    return(0);
}
