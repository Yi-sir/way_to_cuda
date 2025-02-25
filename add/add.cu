#include <stdio.h>

__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = x[tid] + y[tid];
    }
}

int main() {
    int N = 1e7;
    size_t mem_size = sizeof(float) * N;

    float* x, *y, *out;
    float* cuda_x, *cuda_y, *cuda_out;

    x = static_cast<float*>(malloc(mem_size));
    y = static_cast<float*>(malloc(mem_size));

    for(int i = 0; i < N; ++i) {
        x[i] = 1.0;
        y[i] = 2.0;
    }

    cudaMalloc((void**)&cuda_x, mem_size);
    cudaMalloc((void**)&cuda_y, mem_size);
    cudaMalloc((void**)&cuda_out, mem_size);

    cudaMemcpy(cuda_x, x, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_y, y, mem_size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size) / block_size;

    add_kernel<<<grid_size, block_size>>>(cuda_x, cuda_y, cuda_out, N);

    out = static_cast<float*>(malloc(mem_size));
    cudaMemcpy(out, cuda_out, mem_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for(int i = 0; i < 10; ++i){
        printf("out[%d] = %.3f\n", i, out[i]);
    }

    cudaFree(cuda_x);
    cudaFree(cuda_y);
    cudaFree(cuda_out);

    free(x);
    free(y);
    free(out);

    return 0;
}