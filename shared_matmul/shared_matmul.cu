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

template <const int BLOCK_SIZE>
__global__ void sgemm_shared_mem_kernel(float* A, float* B, float* C, int M,
                                        int N, int K) {
  // 当前block计算的是C上坐标为[c_row, c_col]的block
  const uint c_row = blockIdx.x;
  const uint c_col = blockIdx.y;

  // 线程块内共享
  __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE];

  // 当前thread计算的是C上当前block内坐标为[thread_row, thread_col]的值
  const uint thread_row = threadIdx.x / BLOCK_SIZE;
  const uint thread_col = threadIdx.x % BLOCK_SIZE;

  // A: 指针移动到目标block对应行的开头
  A += c_row * BLOCK_SIZE * K;
  // B: 指针移动到目标block对应列的开头
  B += c_col * BLOCK_SIZE;
  // C: 指针移动到目标block的开头
  C += c_row * BLOCK_SIZE * N + c_col * BLOCK_SIZE;

  float tmp = 0.0f;
  // 总共计算K次乘法，循环里每次计算BLOCK_SIZE个
  for (int i = 0; i < K; i += BLOCK_SIZE) {
    // 此时，A已经移动到了某个BLOCK的开头
    // 这里是每个线程都在搬数据
    // 所有线程的工作结果是：把A上一个BLOCK的内容搬到A_shared上
    A_shared[thread_row * BLOCK_SIZE + thread_col] =
        A[thread_row * K + thread_col];
    // B 同理
    B_shared[thread_row * BLOCK_SIZE + thread_col] =
        B[thread_row * N + thread_col];

    __syncthreads();

    // 等待每个线程都搬完数据之后，每个线程开始算自己负责的点[c_row, c_col]
    // 每次需要取BLOCK中一行/一列的内容，算BLOCK_SIZE次乘法
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      tmp += A_shared[thread_row * BLOCK_SIZE + j] *
             B_shared[j * BLOCK_SIZE + thread_col];
    }

    __syncthreads();

    // 一个BLOCK计算结束，BLOCK内每个坐标的结果都计算了一部分
    // 挪到下一个BLOCK继续
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;
  }
  // C的指针已经移动到了目标block的开头
  // 所以这里计算的坐标就是实际坐标
  C[thread_row * N + thread_col] = tmp;
}

void run_sgemm_shared_memory(float* A, float* B, float* C, int m, int n,
                             int k) {
  const int BLOCKSIZE = 32;
  dim3 block_size(BLOCKSIZE * BLOCKSIZE);
  dim3 grid_size(CEIL_DIV(m, BLOCKSIZE), CEIL_DIV(n, BLOCKSIZE));
  sgemm_shared_mem_kernel<BLOCKSIZE>
      <<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void randomize_matrix(float* mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = rand() % 100;
  }
}

int main() {
  int m = 256, n = 256, k = 256;

  float* A = new float[m * k];
  float* B = new float[k * n];
  float* C = new float[m * n];

  float* C_ref = new float[m * n];

  randomize_matrix(A, m * k);
  randomize_matrix(B, k * n);

  float *d_A, *d_B, *d_C;

  cudaMalloc((void**)&d_A, m * k * sizeof(float));
  cudaMalloc((void**)&d_B, n * k * sizeof(float));
  cudaMalloc((void**)&d_C, m * n * sizeof(float));

  cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

  run_sgemm_shared_memory(d_A, d_B, d_C, m, n, k);

  cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  sgemm_naive_cpu(A, B, C_ref, m, n, k);

  for (int i = 0; i < m * n; ++i) {
    if (C[i] != C_ref[i]) {
      std::cout << "Mismatch! [" << i / n << ", " << i % n << "]" << std::endl;
    }
  }

  delete A;
  delete B;
  delete C;
  delete C_ref;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  std::cout << "Success" << std::endl;

  return 0;
}