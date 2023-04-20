#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "../include/helper_cuda.h"

#define NPROBS 3

#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

// #include "../solutions/lab2_sol.cu"
// #define RUN_SOLUTIONS

#ifndef SOLUTION_OF_P1
#define SOLUTION_OF_P1 { }
#endif

#ifndef SOLUTION_OF_P2
#define SOLUTION_OF_P2 { }
#endif

#ifndef SOLUTION_OF_P3
#define SOLUTION_OF_P3 { }
#endif

int main(void)
{
  float errors[NPROBS], GPUtimes[NPROBS], CPUtimes[NPROBS];
  for (int i=0; i<NPROBS; i++) {
    errors[i] = 1<<30;
    GPUtimes[i] = 0;
    CPUtimes[i] = 0;
  }

  TIMER_DEF;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  srand((unsigned int)time(NULL));

// ========================================= Problem 1 ============================================
  {
      // ------------------- not GPU problem ----------------------
      int k = (1<<15), N = 1024*k + 11;
      float *a, *check_vec, alpha = 0.5;
      a = (float*)malloc(N*sizeof(float));
      check_vec = (float*)malloc(N*sizeof(float));

      for (int i = 0; i < N; i++) {
        a[i] = ((float)rand())/((float)RAND_MAX);
        check_vec[i] = 0.0f;
      }

      TIMER_START;
      for (int i=0; i<N; i++)
        check_vec[i] = alpha*a[i]*a[i];
      TIMER_STOP;

      // ---------------- allocing GPU vectors -------------------
      float *b, *dev_a, *dev_b;

      b = (float*)malloc(N*sizeof(float));
      for (int i = 0; i < N; i++)
        b[i] = 0.0f;

      // --------- copy date from host to device -----------

      checkCudaErrors( cudaMalloc(&dev_a, N*sizeof(float)) );
      checkCudaErrors( cudaMalloc(&dev_b, N*sizeof(float)) );

      checkCudaErrors( cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemset(dev_b, 0, N*sizeof(float)) );

      // ------------------ Problem1 solution -----------------
      cudaEventRecord(start, 0);
#ifdef RUN_SOLUTIONS
      SOLUTION_OF_P1
#else
            /* |========================================| */
            /* | Put here your code for solve Problem 1 | */
            /* |========================================| */




#endif
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      // --------- copy results from device to host -----------

      checkCudaErrors( cudaMemcpy(b, dev_b, N*sizeof(float), cudaMemcpyDeviceToHost) );

      // --------------- free GPU variable --------------------

      checkCudaErrors( cudaFree(dev_a) );
      checkCudaErrors( cudaFree(dev_b) );

      // ------------------------------------------------------


      // Compare GPU and not-GPU solution

      float error = 0.0f;
      for (int i = 0; i < N; i++)
        error += abs(check_vec[i] - b[i]);
      errors[0] = error;
      cudaEventElapsedTime(&(GPUtimes[0]), start, stop);
      CPUtimes[0] = TIMER_ELAPSED;

      free(check_vec);
      free(a);
      free(b);
  }
// ========================================= Problem 2 ============================================
  {
      // ------------------- not GPU problem ----------------------
      int n = 120, m = 1000;
      float A[n][m], check_mat[n][m], alpha = 0.5, B[n][m], beta = 0.25;

      for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
          A[i][j] = ((float)rand())/((float)RAND_MAX);
          B[i][j] = ((float)rand())/((float)RAND_MAX);
          check_mat[i][j] = 0.0f;
        }
      }

      TIMER_START;
      for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            check_mat[i][j] = alpha*A[i][j] + beta*B[i][j];
        }
      }
      TIMER_STOP;

      // ---------------- allocing GPU vectors -------------------
      float C[n][m], *dev_A, *dev_B, *dev_C;

      for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
          C[i][j] = 0.0f;
        }
      }

      // --------- copy date from host to device -----------

      checkCudaErrors( cudaMalloc(&dev_A, n*m*sizeof(float)) );
      checkCudaErrors( cudaMalloc(&dev_B, n*m*sizeof(float)) );
      checkCudaErrors( cudaMalloc(&dev_C, n*m*sizeof(float)) );

      checkCudaErrors( cudaMemcpy(dev_A, A, n*m*sizeof(float), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(dev_B, B, n*m*sizeof(float), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemset(dev_C, 0, n*m*sizeof(float)) );


      // ------------------ Problem2 solution -----------------
      cudaEventRecord(start, 0);
#ifdef RUN_SOLUTIONS
      SOLUTION_OF_P2
#else
            /* |========================================| */
            /* | Put here your code for solve Problem 2 | */
            /* |========================================| */




#endif
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      // --------- copy results from device to host -----------

      checkCudaErrors( cudaMemcpy(C, dev_C, n*m*sizeof(float), cudaMemcpyDeviceToHost) );

      // --------------- free GPU variable --------------------

      checkCudaErrors( cudaFree(dev_A) );
      checkCudaErrors( cudaFree(dev_B) );
      checkCudaErrors( cudaFree(dev_C) );

      // ------------------------------------------------------


      // Compare GPU and not-GPU solution

      float error = 0.0f;
      for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
          error += abs(check_mat[i][j] - C[i][j]);
        }
      }
      errors[1] = error;
      cudaEventElapsedTime(&(GPUtimes[1]), start, stop);
      CPUtimes[1] = TIMER_ELAPSED;
  }
// ========================================= Problem 3 ============================================
  {
      // ------------------- not GPU problem ----------------------
      int n = 120, m = 1000, k = 4;
      float A[n][m][k], check_mat[n][m][k], alpha = 0.5, B[n][m][k], beta = 0.25;

      for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
          for (int h=0; h<k; h++) {
            A[i][j][h] = ((float)rand())/((float)RAND_MAX);
            B[i][j][h] = ((float)rand())/((float)RAND_MAX);
            check_mat[i][j][h] = 0.0f;
          }
        }
      }

      TIMER_START;
      for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
          for (int h=0; h<k; h++) {
            check_mat[i][j][h] = alpha*A[i][j][h] + beta*B[i][j][h];
          }
        }
      }
      TIMER_STOP;

      // ---------------- allocing GPU vectors -------------------
      float C[n][m][k], *dev_A, *dev_B, *dev_C;

      for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
          for (int h=0; h<k; h++) {
            C[i][j][h] = 0.0f;
          }
        }
      }

      // --------- copy date from host to device -----------

      checkCudaErrors( cudaMalloc(&dev_A, n*m*k*sizeof(float)) );
      checkCudaErrors( cudaMalloc(&dev_B, n*m*k*sizeof(float)) );
      checkCudaErrors( cudaMalloc(&dev_C, n*m*k*sizeof(float)) );

      checkCudaErrors( cudaMemcpy(dev_A, A, n*m*k*sizeof(float), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(dev_B, B, n*m*k*sizeof(float), cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemset(dev_C, 0, n*m*k*sizeof(float)) );


      // ------------------ Problem3 solution -----------------
      cudaEventRecord(start, 0);
#ifdef RUN_SOLUTIONS
      SOLUTION_OF_P3
#else
            /* |========================================| */
            /* | Put here your code for solve Problem 3 | */
            /* |========================================| */




#endif
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      // --------- copy results from device to host -----------

      checkCudaErrors( cudaMemcpy(C, dev_C, n*m*k*sizeof(float), cudaMemcpyDeviceToHost) );

      // --------------- free GPU variable --------------------

      checkCudaErrors( cudaFree(dev_A) );
      checkCudaErrors( cudaFree(dev_B) );
      checkCudaErrors( cudaFree(dev_C) );

      // ------------------------------------------------------


      // Compare GPU and not-GPU solution

      float error = 0.0f;
      for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
          for (int h=0; h<k; h++) {
            error += abs(check_mat[i][j][h] - C[i][j][h]);
          }
        }
      }
      errors[2] = error;
      cudaEventElapsedTime(&(GPUtimes[2]), start, stop);
      CPUtimes[2] = TIMER_ELAPSED;
  }
// ================================================================================================

#ifdef RUN_SOLUTIONS
  printf("================================= Times and results of SOLUTIONS =================================\n");
#else
  printf("================================== Times and results of my code ==================================\n");
#endif
  for (int i=0; i<NPROBS; i++) {
    printf("Problem %i error = %17.6f --->\t", i, errors[i]);
    (errors[i] == 0) ? printf("\x1B[32mDONE!\x1B[37m\t") : printf("\x1B[31mERROR!\x1B[37m\t");
    printf("GPUtime: %f, \tCPUtime: %f\n", GPUtimes[i], CPUtimes[i]);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return(0);
}
