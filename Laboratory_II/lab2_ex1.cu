#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "../include/helper_cuda.h"
#include <cuda_runtime.h>

#define NPROBS 3

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

#define DBG_CHECK { printf("DBG_CHECK: file %s at line %d\n", __FILE__, __LINE__ ); }
// #define DEBUG
// #define BLK_DISPACH

#define STR(s) #s
#define XSTR(s) STR(s)
#define dtype float

// #include "../solutions/lab2_sol.cu"
#define RUN_SOLUTIONS

#define BLK_SIZE 32
#define GRD_SIZE 2

__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

__global__
void example_kernel(int n, dtype *a, dtype* b, dtype* c)
{
  if (threadIdx.x==0)
      printf("block %d runs on sm %d\n", blockIdx.x, get_smid());

  // [ ... ]
}


#ifdef RUN_SOLUTIONS

#else
            /* |========================================| */
            /* |         Put here your kernels          | */
            /* |========================================| */






#endif



int main(int argc, char *argv[]) {

  // ======================================== Get the device properties ========================================
  printf("======================================= Device properties ========================================\n");

  /* The GPU memory bandwidth is the number of bytes per second which can be transferred between host and device memory.
   * So, you can obtain it by multiplying the number of exchanges done in one second by the bytes exchanged in a single operation.
   * Use the "cudaDeviceProp" structure to compute it by remembering that:
   *    1) The number of operations done in one second is given by doubling the memory clock rate
   *    2) The total amount of bytes exchanged (i.e. the memory bus width) is given by the number of memory controllers
   *        by the width of a single memory controller (that is usually expressed in bits, so divide it by 8 to have the bytes).
   *
   * Take as an example "deviceQuery.cpp" in "https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/".
   *
   * Once you complete the exercise, compare his bandwidth with the theoretical maximum computed here.
   */

#ifdef RUN_SOLUTIONS

#else
            /* |========================================| */
            /* |           Put here your code           | */
            /* |========================================| */




#endif

  printf("====================================== Problem computations ======================================\n");
// =========================================== Set-up the problem ============================================

  if (argc < 2) {
    printf("Usage: lab1_ex2 n\n");
    return(1);
  }
  printf("argv[1] = %s\n", argv[1]);

  // ---------------- set-up the problem size -------------------

  int n = atoi(argv[1]), len = (1<<n), i;

  printf("n = %d --> len = 2^(n) = %d\n", n, len);
  printf("dtype = %s\n", XSTR(dtype));


  // ------------------ set-up the timers ---------------------

  TIMER_DEF;
  const char* lables[NPROBS] = {"CPU", "GPU Layout 1", "GPU Layout 2"};
  float errors[NPROBS], Times[NPROBS], error, MemTime = 0.0;
  long int bytes_moved = 0;
  for (i=0; i<NPROBS; i++) {
    errors[i] = 1<<30;
    Times[i] = 0;
  }


  // ------------------- set-up the problem -------------------

  dtype *a, *b, *CPU_c, *GPU_c;
  a = (dtype*)malloc(sizeof(dtype)*len);
  b = (dtype*)malloc(sizeof(dtype)*len);
  CPU_c = (dtype*)malloc(sizeof(dtype)*len);
  GPU_c = (dtype*)malloc(sizeof(dtype)*len);
  time_t t;
  srand((unsigned) time(&t));

  int typ = (strcmp( XSTR(dtype) ,"int")==0);
  if (typ) {
      // here we generate random ints
      int rand_range = (1<<11);
      printf("rand_range= %d\n", rand_range);
      for (i=0; i<len; i++) {
          a[i] = rand()/(rand_range);
          b[i] = rand()/(rand_range);
          GPU_c[i] = (dtype)0;
      }
  } else {
      // here we generate random floats
      for (i=0; i<len; i++) {
        a[i] = (dtype)rand()/((dtype)RAND_MAX);
        b[i] = (dtype)rand()/((dtype)RAND_MAX);
        GPU_c[i] = (dtype)0;
      }
  }


// ======================================== Running the computations =========================================

  /* Write two different cuda kernels that perform the vector addition with different memory access layouts.
   *  Let k be the total amount of defined threads (i.e. the blockDim*gridDim) and n the vector length; the
   *  first kernel will access in the following way:
   *
   *     th0   th1   th2   ...   ...   thk   th0   th1   ...
   *      |     |     |                 |     |     |
   *      |     |     |                 |     |     |
   *    -----------------------------------------------------------------------
   *   |  0  |  1  |  2  | ... | ... |  k  | k+1 | k+2 | ... |     | ... |  n  |
   *    -----------------------------------------------------------------------
   *
   *
   *  Instead, the second kernel will access in this way:
   *
   *
   *     th0   ...   ...   th0   th1   ...         ...     thk     ...    ...  thk
   *      |                 |     |                         |                   |
   *      |                 |     |                         |                   |
   *    ---------------------------------------------------------------------------
   *   |  0  | ... | ... | n/k | ... |     |     | ... | n-(n/k) | ... | ... |  n  |
   *    ---------------------------------------------------------------------------
   *
   *
   * Find two block_size/grid_size such that:
   *    1) All the operations are performed by a single-stream multiprocessor
   *    2) All your stream multiprocessors compute some operations
   *
   * Check this by using "uint get_smid" as in the example_kernel
   */

  // ========================== CPU computation =========================
  TIMER_START;
  for (i=0; i<len; i++)
    CPU_c[i] = a[i] + b[i];
  TIMER_STOP;
  errors[0] = 0.0;
  Times[0] = TIMER_ELAPSED;



  // ================== GPU computation with Layout 1 ===================

  // ---------------- allocing GPU vectors -------------------
  dtype *dev_a, *dev_b, *dev_c;

  checkCudaErrors( cudaMalloc(&dev_a, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_b, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_c, len*sizeof(dtype)) );


#ifdef RUN_SOLUTIONS

#else
            /* |========================================| */
            /* | Put here your code for solve Problem 1 | */
            /* |========================================| */

  // ------------ copy date from host to device --------------



  // ------------ computation solution with Layout 1 -----------
  TIMER_START;


  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  Times[1] += TIMER_ELAPSED;
  // ----------- copy results from device to host ------------



#endif

  // ------------- Compare GPU and CPU solution --------------

  error = 0.0f;
  for (i = 0; i < len; i++)
    error += (float)fabs(CPU_c[i] - GPU_c[i]);
  errors[1] = error;


  // ================== GPU computation with Layout 2 ===================


#ifdef RUN_SOLUTIONS


#else
            /* |========================================| */
            /* | Put here your code for solve Problem 2 | */
            /* |========================================| */

  // ---------------- Reset the memory in dev_c ----------------



  // ------------ computation solution with Layout 2 -----------
  TIMER_START;



  checkCudaErrors( cudaDeviceSynchronize() );
  TIMER_STOP;
  Times[2] += TIMER_ELAPSED;
  // ------------ copy results from device to host -------------



#endif


  // ------------- Compare GPU and CPU solution --------------

  error = 0.0f;
  for (i = 0; i < len; i++)
    error += (float)fabs(CPU_c[i] - GPU_c[i]);
  errors[2] = error;

  // ----------------- free GPU variable ---------------------

  checkCudaErrors( cudaFree(dev_a) );
  checkCudaErrors( cudaFree(dev_b) );
  checkCudaErrors( cudaFree(dev_c) );

  // ---------------------------------------------------------


// ============================================ Print the results ============================================

#ifdef RUN_SOLUTIONS
  printf("================================= Times and results of SOLUTIONS =================================\n");
#else
  printf("================================== Times and results of my code ==================================\n");
#endif
  printf("Solution type\terror\ttime\n");
  for (int i=0; i<NPROBS; i++) {
    printf("%12s:\t%5.3f\t%5.3f\n", lables[i], errors[i], Times[i]);
  }
  printf("\n");

#ifdef RUN_SOLUTIONS


#else
  // Print here your Memory Bandwidth an the Throughput of both the GPU computations



#endif


// ==================================== Iterate over the kernel dimensions ====================================

  /*  Now, iterate over different block and grid sizes to find the configuration with better performance. The
   *   idea is to compute something similar to this:
   *
   *  Layout1 times:
   *  blk_size\grd_size:      1      3      7     14     28     56
   *                 32:      X      X      X      X      X      X
   *                 64:      X      X      X      X      X      X
   *                128:      X      X      X      X      X      X
   *                256:      X      X      X      X      X      X
   *                512:      X      X      X      X      X      X
   *               1024:      X      X      X      X      X      X
   *
   *  Note: since we are here not interested in the memory performance and checking again if the results are
   *    correct, copy the vectors a and b on the GPU only the first time and don't copy back the results of c.
   */



  checkCudaErrors( cudaMalloc(&dev_a, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_b, len*sizeof(dtype)) );
  checkCudaErrors( cudaMalloc(&dev_c, len*sizeof(dtype)) );

  checkCudaErrors( cudaMemcpy(dev_a, a, len*sizeof(dtype), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(dev_b, b, len*sizeof(dtype), cudaMemcpyHostToDevice) );

#ifdef RUN_SOLUTIONS



#else
          /* |========================================| */
          /* |           Put here your code           | */
          /* |========================================| */




#endif


  checkCudaErrors( cudaFree(dev_a) );
  checkCudaErrors( cudaFree(dev_b) );
  checkCudaErrors( cudaFree(dev_c) );

  free(a);
  free(b);
  free(CPU_c);
  free(GPU_c);

  return(0);
}
