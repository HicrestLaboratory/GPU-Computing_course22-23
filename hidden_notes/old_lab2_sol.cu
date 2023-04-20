__global__
void Problem_1_solution(int n, float *a, float alpha, float* b)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i<n)
    b[i] = alpha*a[i]*a[i];
}

__global__
void Problem_2_solution(int n, int m, float *A, float *B, float alpha, float beta, float *C)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i<n && j<m)
    C[i * m + j] = alpha*A[i * m + j] + beta*B[i * m + j];
}

__global__
void Problem_3_solution(int n, int m, int k, float *A, float *B, float alpha, float beta, float *C)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int h = blockIdx.z*blockDim.z + threadIdx.z;

  if (i<n && j<m && h<k)
    C[h * n * m + i * m + j] = alpha*A[h * n * m + i * m + j] + beta*B[h * n * m + i * m + j];
}

#define SOLUTION_OF_P1 { \
  dim3 block_size(1024, 1, 1);  \
  dim3 grid_size((N/1024) + ((N%1024)!=0), 1, 1);  \
  \
  Problem_1_solution<<<grid_size, block_size>>>(N, dev_a, alpha, dev_b);  \
}

#define SOLUTION_OF_P2 {  \
  dim3 block_size(32, 32, 1);  \
  dim3 grid_size((n/32) + ((n%32)!=0), (m/32) + ((m%32)!=0), 1);  \
  \
  Problem_2_solution<<<grid_size, block_size>>>(n, m, dev_A, dev_B, alpha, beta, dev_C);  \
}

#define SOLUTION_OF_P3 {  \
  dim3 block_size(16, 16, 4);  \
  dim3 grid_size((n/16) + ((n%16)!=0), (m/16) + ((m%16)!=0), 1);  \
  \
  Problem_3_solution<<<grid_size, block_size>>>(n, m, k, dev_A, dev_B, alpha, beta, dev_C);  \
}
