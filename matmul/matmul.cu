#include <stdio.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// blockIdx.x: grid中，当前block的x坐标
// blockDim.x: grid中，一个block的x方向长度（thread数量）

// A: M * K
// B: K * N

// 二维的grid，二维的block
// 是为了方便算矩阵元素的坐标？
__global__ void sgemm_naive_kernel(float* A, float* B, float* C, int M, int N, int K) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < M) {
        float sum = 0.0f;
        for(int i = 0; i < K; ++ i) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
}

void sgemm_naive_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for(int x = 0; x < M; ++x) {
        for(int y = 0; y < N; ++y) {
            float sum = 0.0f;
            for(int i = 0; i < K; ++i) {
                sum += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = sum;
        }
    }
}

// 每个block有32*32个线程
// 需要开多少个block，视m和n而定
void run_sgemm_naive(float* A, float* B, float* C, int m, int n, int k) {
    dim3 block_size(32, 32);
    dim3 grid_size(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    sgemm_naive_kernel<<<grid_size, block_size>>>(A, B, C, m, n, k);
}

void randomize_matrix(float* mat, int N) {
    for(int i = 0; i < N; ++i) {
        mat[i] = rand() % 100;
    }
}

int main() {
    int m = 256, n = 256, k = 256;

    float* A, *B, *C, *C_ref;
    float* d_A, *d_B, *d_C;

    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];
    C_ref = new float[m * n];

    randomize_matrix(A, m * k);
    randomize_matrix(B, k * n);

    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, n * k * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    run_sgemm_naive(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    sgemm_naive_cpu(A, B, C_ref, m, n, k);

    for (int i = 0; i < m * n; i++)
    {
        if (C[i] != C_ref[i])
        {
            printf("Error: mismatch at index %d, expected %f, got %f\n", i, C_ref[i], C[i]);
            return 1;
        }
    }

    delete A;
    delete B;
    delete C;
    delete C_ref;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("success!\n");
    return 0;
}