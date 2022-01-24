#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    return result;
}


//////////////////
// ALGORITHM #1 //
//////////////////


__global__
void GEMMKernel(double* A,
		double* B,
		double* C,
		double  alpha,
		double  beta,
		int     M,
		int     N,
		int     K) {

     int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
     int tIDy = blockIdx.y * blockDim.y + threadIdx.y;

     if (tIDx < M and tIDy < N) {
     	double Cvalue = beta * C[tIDx + M * tIDy];
     	for (int idx = 0; idx < K; ++idx) {
	    Cvalue += alpha * A[tIDx + M * idx] * B[idx + K * tIDy];
     	}
	C[tIDx + M * tIDy] = Cvalue;
     }
}


//////////////////
// ALGORITHM #2 //
//////////////////


__global__
void GEMMSharedKernel(double* A,
		      double* B,
		      double* C,
		      double  alpha,
		      double  beta,
		      int     M,
		      int     N,
		      int     K) {

     const int BLOCK_SIZE = 32;

     // Block row and column
     int x = blockIdx.x * blockDim.x;
     int y = blockIdx.y * blockDim.y;
     double Cvalue = 0.0;
     int tIDx = x + threadIdx.x;
     int tIDy = y + threadIdx.y;
     int iters = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

     for (int iter = 0; iter < iters; ++iter) {
       __shared__ double As[BLOCK_SIZE][BLOCK_SIZE + 1];
     	 __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

       int id = BLOCK_SIZE * iter;
       if (tIDx < M and id + threadIdx.y < K) {
         As[threadIdx.x][threadIdx.y] = A[M * (id + threadIdx.y) + tIDx];
       }
       if (tIDy < N and id + threadIdx.x < K) {
         Bs[threadIdx.x][threadIdx.y] = B[K * tIDy + id + threadIdx.x];
       }

       __syncthreads();

       for (int i = 0; i < BLOCK_SIZE; ++i) {
         if (id + i < K) {
           Cvalue += As[threadIdx.x][i]* Bs[i][threadIdx.y];
         }
       }
     }

     if (tIDx < M and tIDy < N) {
       C[tIDx + M * tIDy] *= beta;
       C[tIDx + M * tIDy] += alpha * Cvalue;
     }
}




//////////////////
// ALGORITHM #3 //
//////////////////
/*

Refer to Volkov and Demmel's paper on Supercomputing
Vector length: 64 //stripmined into two warps by GPU
Registers: a, c[1:16] //each is 64-element vector
Shared memory: b[16][16] //may include padding

Compute pointers in A, B and C using thread ID
c[1:8] = 0
do
	b[1:16][1:16] = next 1616 block in B or tr(B)
	local barrier //wait until b[][] is written by all warps

	unroll for i = 1 to 8 do
	       a = next column of A
	       c[1] += a*b[i][1]
	       c[2] += a*b[i][2]
	       c[3] += a*b[i][3]
	       ...
	       c[8] += a*b[i][8]
	endfor

	// rank-1 update of C’ s block
	// data parallelism = 1024
	// stripmined in software
	// into 16 operations
	// access to b[][] is stride-1

	local barrier //wait until done using b[][]

	update pointers in A and B
repeat until pointer in B is out of range

Merge c[1:8] with 64x16 block of C in memory

*/

template <int BLOCK_SIZE, int SIZE_X, int SIZE_Y>
__global__
void GEMMAPlusKernel(double* A,
                     double* B,
                     double* C,
                     double  alpha,
                     double  beta,
                     int     M,
                     int     N,
                     int     K) {

    int threadMoved = BLOCK_SIZE * threadIdx.y + threadIdx.x;
    int M_0         = blockIdx.y * SIZE_X + threadMoved;
    int N_0[2]      = {threadIdx.y + blockIdx.x * SIZE_Y,
                       threadIdx.y + blockIdx.x * SIZE_Y + BLOCK_SIZE};

    double a[BLOCK_SIZE];
    double c[SIZE_Y];
    __shared__ double Bs[BLOCK_SIZE * SIZE_Y];

    // Compute pointers in A, B, C using thread ID
    for (int id = 0; id < SIZE_Y; ++id) {
      c[id] = 0.0;
    }

    for (int i = 0; i < K; i += BLOCK_SIZE) {
      if (M_0 < M) {
        a[0] = A[M_0 + M * i];
      }
      // rank - 1 uodate of C's block
      // data parallelism = 1024
      // stripmined in software
      // into 16 operations
      // access to b[][] is stride - 1
      for (int j = 1; j < BLOCK_SIZE; ++j) {
        if (M_0 < M and i + j < K) {
          a[j] = A[M_0 + (i + j) * M];
        }
      }

      const int moved = threadIdx.x + i;
      __syncthreads();
      if (moved < K and N_0[0] < N) {
        Bs[threadMoved] = B[N_0[0] * K + moved];
      }
      if (moved < K and N_0[1] < N) {
        Bs[threadMoved + SIZE_X] = B[N_0[1] * K + moved];
      }
      __syncthreads();


      for(int u = 0; u < BLOCK_SIZE; ++u){
        for(int v = 0; v < SIZE_Y; ++v) {
          if(u + i < K) {
            c[v] += Bs[u + BLOCK_SIZE * v] * a[u];
	  }
	}
      }
    }

    // Merge c[1:8] with 64 x 16 block of C in memory
    for(int i = 0; i < SIZE_Y; i++){
      if(M_0 < M and blockIdx.x * SIZE_Y + i < N){
        double value = beta * C[M_0 + (blockIdx.x * SIZE_Y + i) * M];
        value += alpha * c[i];
        C[M_0 + (blockIdx.x * SIZE_Y + i) * M] = value;
      }
    }
}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* A,
           double* B,
           double* C,
           double* alpha,
           double* beta,
           int     M,
           int     N,
           int     K) {

    const int BLOCK_SIZE = 8;
    const int SIZE_X = 64;
    const int SIZE_Y = 16;

    // Dimensions of a grid
    const int threads_per_block_x = BLOCK_SIZE;
    const int threads_per_block_y = BLOCK_SIZE;
    dim3 block_size(threads_per_block_x, threads_per_block_y);

    int blocks_per_grid_y = (M + SIZE_X - 1) / SIZE_X;
    int blocks_per_grid_x = (N + SIZE_Y - 1) / SIZE_Y;
    dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

//    GEMMKernel<<<grid_size, block_size>>>(A, B, C, *alpha, *beta, M, N, K);

//    GEMMSharedKernel<<<grid_size, block_size>>>(A, B, C, *alpha, *beta, M, N, K);

    GEMMAPlusKernel<BLOCK_SIZE, SIZE_X, SIZE_Y>
    		   <<<grid_size, block_size>>>(A, B, C, *alpha, *beta, M, N, K);

    return 0;
}




////////////////////////////////////////////////////////////////////////////////
//                      GEMMSigmoid Calculation                               //
////////////////////////////////////////////////////////////////////////////////


template <int BLOCK_SIZE, int SIZE_X, int SIZE_Y>
__global__
void GEMMSigmoidKernel(double* A,
                       double* B,
                       double* C,
                       double* D,
                       double  alpha,
                       double  beta,
                       int     M,
                       int     N,
                       int     K) {

    int threadMoved = BLOCK_SIZE * threadIdx.y + threadIdx.x;
    int M_0         = blockIdx.y * SIZE_X + threadMoved;
    int N_0[2]      = {threadIdx.y + blockIdx.x * SIZE_Y,
                       threadIdx.y + blockIdx.x * SIZE_Y + BLOCK_SIZE};

    double a[BLOCK_SIZE];
    double c[SIZE_Y];
    __shared__ double Bs[BLOCK_SIZE * SIZE_Y];

    // Compute pointers in A, B, C using thread ID
    #pragma unroll
    for (int id = 0; id < SIZE_Y; ++id) {
      c[id] = 0.0;
    }

    for (int i = 0; i < K; i += BLOCK_SIZE) {
      if (M_0 < M) {
        a[0] = A[M_0 + M * i];
      }

      for (int j = 1; j < BLOCK_SIZE; ++j) {
        if (M_0 < M and i + j < K) {
          a[j] = A[M_0 + (i + j) * M];
        }
      }

      const int moved = threadIdx.x + i;
      __syncthreads();
      if (moved < K and N_0[0] < N) {
        Bs[threadMoved] = B[N_0[0] * K + moved];
      }
      if (moved < K and N_0[1] < N) {
        Bs[threadMoved + SIZE_X] = B[N_0[1] * K + moved];
      }
      __syncthreads();


      #pragma unroll
      for(int u = 0; u < BLOCK_SIZE; ++u){
        #pragma unroll
        for(int v = 0; v < SIZE_Y; ++v) {
          if(u + i < K) {
            c[v] += Bs[u + BLOCK_SIZE * v] * a[u];
	  }
	}
      }
    }
    // Merge c[1:8] with 64 x 16 block of C in memory
    #pragma unroll
    for(int i = 0; i < SIZE_Y; i++){
      if(M_0 < M and blockIdx.x * SIZE_Y + i < N){
        double value = beta * C[M_0];
        value += alpha * c[i];
        D[M_0 + (blockIdx.x * SIZE_Y + i) * M] = 1.0 / (1.0 + exp(-1.0*value));
      }
    }
}

/*
Routine to perform an in-place GEMM operation, i.e.,
D := sigmoid(alpha*A*B + beta*C) element-wise, where C is a R-dimensional
column vector
*/
int GEMMSigmoid(double* A,
                double* B,
                double* C,
                double* D,
                double  alpha,
                double  beta,
                int     M,
                int     N,
                int     K) {

    const int BLOCK_SIZE = 8;
    const int SIZE_X = 64;
    const int SIZE_Y = 16;

    // Dimensions of a grid
    const int threads_per_block_x = BLOCK_SIZE;
    const int threads_per_block_y = BLOCK_SIZE;
    dim3 block_size(threads_per_block_x, threads_per_block_y);

    int blocks_per_grid_y = (M + SIZE_X - 1) / SIZE_X;
    int blocks_per_grid_x = (N + SIZE_Y - 1) / SIZE_Y;
    dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

    GEMMSigmoidKernel<BLOCK_SIZE, SIZE_X, SIZE_Y>
    		   <<<grid_size, block_size>>>(A, B, C, D, alpha, beta, M, N, K);

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//                      GEMMAddition Calculation                              //
////////////////////////////////////////////////////////////////////////////////


template <int BLOCK_SIZE, int SIZE_X, int SIZE_Y>
__global__
void GEMMAdditionKernel(double* A,
                        double* B,
                        double* C,
                        double* D,
                        double  alpha,
                        double  beta,
                        int     M,
                        int     N,
                        int     K) {

    int threadMoved = BLOCK_SIZE * threadIdx.y + threadIdx.x;
    int M_0         = blockIdx.y * SIZE_X + threadMoved;
    int N_0[2]      = {threadIdx.y + blockIdx.x * SIZE_Y,
                       threadIdx.y + blockIdx.x * SIZE_Y + BLOCK_SIZE};

    double a[BLOCK_SIZE];
    double c[SIZE_Y];
    __shared__ double Bs[BLOCK_SIZE * SIZE_Y];

    // Compute pointers in A, B, C using thread ID
    #pragma unroll
    for (int id = 0; id < SIZE_Y; ++id) {
      c[id] = 0.0;
    }

    for (int i = 0; i < K; i += BLOCK_SIZE) {
      if (M_0 < M) {
        a[0] = A[M_0 + M * i];
      }

      for (int j = 1; j < BLOCK_SIZE; ++j) {
        if (M_0 < M and i + j < K) {
          a[j] = A[M_0 + (i + j) * M];
        }
      }

      const int moved = threadIdx.x + i;
      __syncthreads();
      if (moved < K and N_0[0] < N) {
        Bs[threadMoved] = B[N_0[0] * K + moved];
      }
      if (moved < K and N_0[1] < N) {
        Bs[threadMoved + SIZE_X] = B[N_0[1] * K + moved];
      }
      __syncthreads();


      #pragma unroll
      for(int u = 0; u < BLOCK_SIZE; ++u){
        #pragma unroll
        for(int v = 0; v < SIZE_Y; ++v) {
          if(u + i < K) {
            c[v] += Bs[u + BLOCK_SIZE * v] * a[u];
	  }
	}
      }
    }
    // Merge c[1:8] with 64 x 16 block of C in memory
    #pragma unroll
    for(int i = 0; i < SIZE_Y; i++){
      if(M_0 < M and blockIdx.x * SIZE_Y + i < N){
        double value = beta * C[M_0];
        value += alpha * c[i];
        D[M_0 + (blockIdx.x * SIZE_Y + i) * M] = value;
      }
    }
}


/*
Routine to perform an in-place GEMM operation, i.e.,
D := (alpha*A*B + beta*C) element-wise, where C is a R-dimensional
column vector
*/
int GEMMAddition(double* A,
                 double* B,
                 double* C,
                 double* D,
                 double  alpha,
                 double  beta,
                 int     M,
                 int     N,
                 int     K) {

    const int BLOCK_SIZE = 8;
    const int SIZE_X = 64;
    const int SIZE_Y = 16;

    // Dimensions of a grid
    const int threads_per_block_x = BLOCK_SIZE;
    const int threads_per_block_y = BLOCK_SIZE;
    dim3 block_size(threads_per_block_x, threads_per_block_y);

    int blocks_per_grid_y = (M + SIZE_X - 1) / SIZE_X;
    int blocks_per_grid_x = (N + SIZE_Y - 1) / SIZE_Y;
    dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

    GEMMAdditionKernel<BLOCK_SIZE, SIZE_X, SIZE_Y>
    		   <<<grid_size, block_size>>>(A, B, C, D, alpha, beta, M, N, K);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//                      GEMMTranspose Calculation                             //
////////////////////////////////////////////////////////////////////////////////

template <int X, int Y>
__global__
void GEMMTransposeKernel(double* A,
                         double* B,
                         double* C,
                         double  alpha,
                         int     M,
                         int     N,
                         int     K) {

  int tIDx = threadIdx.x;
  int bIDx = blockIdx.x;
  int tIDy = threadIdx.y;
  int bIDy = blockIdx.y;

  // Compute pointers in A, B, C using thread ID
  double a[X] = {0};
  double c[Y] = {0};
  __shared__ double Bs[X * Y];
  int it_a = tIDx + tIDy * Y + 64 * bIDy;
  int it_b = tIDx + bIDx * Y;
  int init_b = tIDy;
  int it = -1;

  // rank - 1 uodate of C's block
  // data parallelism = 1024
  // stripmined in software
  // into 16 operations
  // access to b[][] is stride - 1

  for(; init_b < K; init_b += X){
     it = it + 1;

     for(int i = 0; i < 4; ++i){
        if (it_a < M and i < K - it * X)   a[i] = A[it_a + i * M + X * it * M];
     }

     if (it_b < N)   Bs[tIDy + tIDx * X]= B[it_b + init_b * N];
     __syncthreads();
     for (int i = 0; i < 16; ++i){
     	 for (int j = 0; j < 4; ++j) {
	     c[i] += a[j] * Bs[j + i * X];
	 }
     }
     __syncthreads();
  }

  // Merge c with the block of C in memory
  if (it_a < M){
     for (int i = 0; i < 16; ++i) {
     	 if (i < N - X * bIDx) {
	    C[(i + bIDx * Y) * M + it_a]  = c[i] +
	    	   alpha * C[(i + bIDx *Y) * M + it_a];
	 }
     }
  }
}


int GEMMTranspose(double* A,
                  double* B,
                  double* C,
                  double  alpha,
                  int     M,
                  int     N,
                  int     K) {

    const int SIZE_X = 16;
    const int SIZE_Y = 4;

    const int threads_per_block_x = SIZE_X;
    const int threads_per_block_y = SIZE_X * SIZE_Y;
    dim3 block_size(SIZE_X, SIZE_Y);

    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;
    dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

    GEMMTransposeKernel<SIZE_Y, SIZE_X>
                       <<<grid_size, block_size>>>(A, B, C, alpha, M, N, K);

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//                      Hadamard Calculation                                  //
////////////////////////////////////////////////////////////////////////////////

__global__
void HadamardKernel(double* A,
                    double* B,
                    double* C,
                    int     M,
                    int     N,
                    int     K) {

    int tIDx = blockIdx.x * blockDim.x + threadIdx.x;
    int tIDy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIDx < M and tIDy < N) {
       double Cvalue = 0.0;
       double sigm   = C[tIDx + M * tIDy];
       for (int idx = 0; idx < K; ++idx) {
       	   Cvalue += A[idx + K * tIDx] * B[idx + K * tIDy];
       }
       sigm *= (1 - sigm) * Cvalue;
       C[tIDx + M * tIDy] = sigm;
    }
}


/*
Routine to perform an in-place GEMM operation, i.e., C := t(A)*B .* C .* (1 - C)
*/

int Hadamard(double* A,
    	       double* B,
	           double* C,
             int     M,
             int     N,
             int     K) {

    const int threads_per_block_x = 32;   const int threads_per_block_y = 32;
    dim3 block_size(threads_per_block_x, threads_per_block_y);

    int blocks_per_grid_x = (M + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (N + threads_per_block_y - 1) / threads_per_block_y;
    dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

    HadamardKernel<<<grid_size, block_size>>>(A, B, C, M, N, K);

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
//                      Hadamard Calculation                                  //
////////////////////////////////////////////////////////////////////////////////

__global__
void SoftmaxKernel(double* X,
                   double* Y,
                   double* Z,
                   double* A,
                   double  lambda,
                   int     M,
                   int     N) {

     int tIDx = blockDim.x * blockIdx.x + threadIdx.x;
     int tIDy = blockDim.y * blockIdx.y + threadIdx.y;

     if (tIDx < M and tIDy < N) {
       int idx = tIDx + M * tIDy;
       A[idx] = lambda * (exp(X[idx]) / Y[tIDy] - Z[idx]);
     }
}

/*
Function that performs in-place softmax and difference,
A = lambda * (exp(X) / Y - Z) where Y = colsumExp(X).
*/
int Softmax(double* X,
            double* Y,
            double* Z,
            double* A,
            double  lambda,
            int     M,
            int     N) {

      // Block and grid dimensions
      int threads_per_block_x = 32;   int threads_per_block_y = 8;
      dim3 block_size(threads_per_block_x, threads_per_block_y);

      int blocks_per_grid_x = (threads_per_block_x + M - 1) / (threads_per_block_x);
      int blocks_per_grid_y = (threads_per_block_y + N - 1) / (threads_per_block_y);
      dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

      SoftmaxKernel<<<grid_size, block_size>>>(X, Y, Z, A, lambda, M, N);

      return 0;
}


////////////////////////////////////////////////////////////////////////////////
//                   Sum of Exp Columns Calculation                           //
////////////////////////////////////////////////////////////////////////////////

__global__
void SumOfExpColKernel(double* X,
                       double* Y,
                       int     M,
                       int     N) {

      int tID = blockDim.x * blockIdx.x + threadIdx.x;

      if (tID < N) {
      	 Y[tID] = 0.0;
         for (int idx = 0; idx < M; ++idx) {
           Y[tID] += exp(X[idx + M * tID]);
         }
      }
}

int SumOfExpCol(double* X,
                double* Y,
                int     M,
                int     N) {

      int threads_per_block = 32;
      int blocks_per_grid   = (N + threads_per_block - 1) / threads_per_block;

      SumOfExpColKernel<<<blocks_per_grid, threads_per_block>>>(X, Y, M, N);

      return 0;
}



////////////////////////////////////////////////////////////////////////////////
//                   Sum of Rows Calculation                                  //
////////////////////////////////////////////////////////////////////////////////

__global__
void SumOfRowKernel(double* X,
                    double* Y,
                    int     M,
                    int     N) {

    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    if (tID < M) {
      Y[tID] = 0.0;
      for (int idx = 0; idx < N; ++idx) {
        Y[tID] += X[tID + M * idx];
      }
    }
}

int SumOfRow(double* X,
             double* Y,
             int     M,
             int     N) {

    int threads_per_block = 32;
    int blocks_per_grid   = (M + threads_per_block - 1) / threads_per_block;

    SumOfRowKernel<<<blocks_per_grid, threads_per_block>>>(X, Y, M, N);

    return 0;
}
