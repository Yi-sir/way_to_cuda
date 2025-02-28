#include <cuda_runtime.h>

#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// 自己尝试写的，计算结果有误...看来我没懂

/*
A: [M * K]
B: [K * N]
C: [M * N]

Matrix block size: [BM * BK], [BK * BN], [BM * BN],
so cuda grid size is dim3(CEIL_DIV(M, BM), CEIL_DIV(N, BN)).

Cuda shared memory size: same with block size of A and B.

Tiled size: TM in A, and TN in B.

In this case, TM and TN is set {8, 8}
*/

template <int BM, int BK, int BN, int TM, int TN>
__global__ void sgemm_tiled_2d(float* A, float* B, float* C, int M, int N,
                               int K) {
  // 本线程算C上当前block里的某一组点的起始坐标(以TM，TN为单位)
  int block_row_id = threadIdx.y;
  int block_col_id = threadIdx.x;

  // 当前block在C上的位置
  int block_id_x = blockIdx.x;
  int block_id_y = blockIdx.y;

  int global_block_id_x = block_id_x * BM;
  int global_block_id_y = block_id_y * BN;

  // 当前线程计算的block的起始点在C上的位置
  int global_thread_id_x = global_block_id_x + block_row_id * TM;
  int global_thread_id_y = global_block_id_y + block_col_id * TN;

//   int block_size_x = blockDim.x;
//   int block_size_y = blockDim.y;

  __shared__ float A_shared[BM * BK];
  __shared__ float B_shared[BK * BN];

  A += block_id_x * BM * K;
  B += block_id_y * BN;
  C += block_id_x * BM * K + block_id_y * BN;

  // 存放临时结果
  float res[TM * TN] = {0.0f};

  // block从前向后移动（从上向下）
  for (int blk_idx = 0; blk_idx < K; blk_idx += BK) {
    // 搬运内存，A上每个线程需要搬运TM * BK，B上每个线程需要搬运BK * TN
    for (int bk = 0; bk < BK; ++bk) {
      for (int tm = 0; tm < TM; ++tm) {
        // 搬A
        int inner_row = global_thread_id_x + block_row_id * TM + tm;
        int inner_col = blk_idx * BK + bk;
        A_shared[tm * BK + bk] = A[inner_row * K + inner_col];
      }
      for (int tn = 0; tn < TN; ++tn) {
        // 搬B
        int inner_row = blk_idx * BK + bk;
        int inner_col = global_thread_id_y + block_col_id * TN + tn;
        B_shared[bk * TN + tn] = B[inner_row * N + inner_col];
      }
    }

    __syncthreads();

    A += BK;
    B += BK * N;

    for (int m = 0; m < TM; ++m) {
      for (int n = 0; n < TN; ++n) {
        for (int k = 0; k < BK; ++k)
          res[m * TN + n] += A_shared[m * BK + k] * B_shared[k * TN * n];
      }
    }

    __syncthreads();
  }
  // 算完
  for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; ++n) {
      int real_id = (global_thread_id_x + m) * N + global_thread_id_y + n;
      C[real_id] = res[m * TN + n];
    }
  }
}

void run_gemm_tiled_2d(float* A, float* B, float* C, int M, int N, int K) {
  const uint BM = 32;
  const uint BN = 32;
  const uint BK = 32;
  const uint grid_x = CEIL_DIV(M, BM);
  const uint grid_y = CEIL_DIV(N, BN);
  dim3 grid_size = {grid_x, grid_y};
  const uint TM = 8;
  const uint TN = 8;
  const uint block_x = CEIL_DIV(BM, TM);
  const uint block_y = CEIL_DIV(BN, TN);
  dim3 block_size = {block_x, block_y};
  sgemm_tiled_2d<BM, BK, BN, TM, TN>
      <<<grid_size, block_size>>>(A, B, C, M, N, K);
}

void sgemm_naive_cpu(float* A, float* B, float* C, int M, int N, int K) {
  for (int x = 0; x < M; ++x) {
    for (int y = 0; y < N; ++y) {
      float sum = 0.0f;
      for (int i = 0; i < K; ++i) {
        sum += A[x * K + i] * B[i * N + y];
      }
      C[x * N + y] = sum;
    }
  }
}

void randomize_matrix(float* mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = rand() % 100;
  }
}

int main() {
  int m = 256;
  int n = 256;
  int k = 256;

  // Allocate memory for matrices
  float *A, *B, *C, *C_ref;
  float *d_A, *d_B, *d_C;

  A = new float[m * k];
  B = new float[k * n];
  C = new float[m * n];
  // save reference result
  C_ref = new float[m * n];

  // Initialize matrices
  randomize_matrix(A, m * k);
  randomize_matrix(B, k * n);

  // Allocate device memory
  cudaMalloc((void**)&d_A, m * k * sizeof(float));
  cudaMalloc((void**)&d_B, k * n * sizeof(float));
  cudaMalloc((void**)&d_C, m * n * sizeof(float));

  // Copy matrices to device
  cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

  run_gemm_tiled_2d(d_A, d_B, d_C, m, n, k);

  // Copy result to host
  cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  // Run reference sgemm
  sgemm_naive_cpu(A, B, C_ref, m, n, k);

  // Verify result
  for (int i = 0; i < m * n; i++) {
    if (C[i] != C_ref[i]) {
      printf("Error: mismatch at index %d, expected %f, got %f\n", i, C_ref[i],
             C[i]);
      return 1;
    }
  }

  free(A);
  free(B);
  free(C);
  free(C_ref);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  printf("Success!\n");
  return 0;
}