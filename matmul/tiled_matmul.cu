#include <cuda_runtime.h>

#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

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

// A: M * K
// B: K * N

// blockA: BM * BK
// blockB: BK * BN
// blockC: BM * BN

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(float* A, float* B, float* C, int M,
                                            int N, int K) {
  const uint c_row = blockIdx.y;
  const uint c_col = blockIdx.x;

  __shared__ float A_shared[BM * BK];
  __shared__ float B_shared[BK * BN];

  // C上一个block内的坐标，注意thread_row以TM个行为单位
  const uint thread_row = threadIdx.x / BN;
  const uint thread_col = threadIdx.x % BN;

  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * K + c_col * BN;

  // A上当前block对应的起始坐标
  int global_m_pos = c_row * BM * K;
  // B上当前block对应的起始坐标
  int global_n_pos = c_col * BN;
  // 矩阵边界
  const uint m_size = M * K;
  const uint n_size = N * K;

  // 线程数必须和块大小一致，BM == BN ？
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);

  // A上block中的行列
  const uint A_inner_row = threadIdx.x / BK;
  const uint A_inner_col = threadIdx.x % BK;
  // B上block中的行列
  const uint B_inner_row = threadIdx.x / BN;
  const uint B_inner_col = threadIdx.x % BN;

  // 一次计算TM行的结果
  float thread_results[TM] = {0.0};

  for (uint bk_idx = 0; bk_idx < K; bk_idx += BK) {
    A_shared[A_inner_row * BK + A_inner_col] =
        (global_m_pos + A_inner_row * K + A_inner_col < m_size)
            ? A[A_inner_row * K + A_inner_col]
            : 0.0f;
    B_shared[B_inner_row * BN + B_inner_col] =
        (global_n_pos + B_inner_row * N + B_inner_col < n_size)
            ? B[B_inner_row * N + B_inner_col]
            : 0.0f;
    __syncthreads();

    A += BK;
    B += BK * N;
    global_m_pos += BK;
    global_n_pos += BK * N;

    // 一次计算BK个乘法
    // 这里把BK提前了，也就是说一次是A_block中长度为TM的一列数和B_block上的一个值做计算
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
      // 对于当前线程，其要计算的列是确定的，即thread_col
      float tmp_b = B_shared[dot_idx * BN + thread_col];
      // 一个线程要计算TM个行的结果，一个thread_row代表TM个行
      for (uint res_idx = 0; res_idx < TM; ++res_idx) {
        thread_results[res_idx] +=
            A_shared[(thread_row * TM + res_idx) * BK + dot_idx] * tmp_b;
      }
    }
    __syncthreads();
  }
  for (uint res_idx = 0; res_idx < TM; res_idx++) {
    if (c_row * BM + thread_row * TM + res_idx < M &&
        c_col * BN + thread_col < N) {
      C[(thread_row * TM + res_idx) * N + thread_col] = thread_results[res_idx];
    }
  }
}

void run_sgemm_blocktiling_1d(float* A, float* B, float* C, int m, int n,
                              int k) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 grid_size(CEIL_DIV(n, BN), CEIL_DIV(m, BM));
  dim3 block_size((BM * BN) / TM);
  sgemm_blocktiling_1d_kernel<BM, BN, BK, TM>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
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
  float* A, *B, *C, *C_ref;
  float* d_A, *d_B, *d_C;

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

  run_sgemm_blocktiling_1d(d_A, d_B, d_C, m, n, k);

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