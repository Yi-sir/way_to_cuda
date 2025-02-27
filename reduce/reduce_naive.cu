#include <cuda_runtime.h>

#include <iostream>

// ==1960493== NVPROF is profiling process 1960493, command: ./reduce
// sum = -16777216
// success
// ==1960493== Profiling application: ./reduce
// ==1960493== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   48.48%  67.268ms         1  67.268ms  67.268ms  67.268ms  [CUDA memcpy DtoH]
//                    36.26%  50.311ms         1  50.311ms  50.311ms  50.311ms  void reduce_naive_kernel<int=256>(int*, int*, int)
//                    15.26%  21.169ms         1  21.169ms  21.169ms  21.169ms  [CUDA memcpy HtoD]
//       API calls:   61.62%  225.74ms         2  112.87ms  75.211us  225.66ms  cudaMalloc
//                    38.19%  139.91ms         2  69.954ms  21.280ms  118.63ms  cudaMemcpy
//                     0.10%  354.38us         2  177.19us  118.30us  236.08us  cudaFree
//                     0.07%  261.89us        97  2.6990us     333ns  103.22us  cuDeviceGetAttribute
//                     0.01%  36.683us         1  36.683us  36.683us  36.683us  cudaLaunchKernel
//                     0.01%  21.309us         1  21.309us  21.309us  21.309us  cuDeviceGetName
//                     0.00%  13.365us         1  13.365us  13.365us  13.365us  cuDeviceGetPCIBusId
//                     0.00%  3.0120us         3  1.0040us     346ns  2.1590us  cuDeviceGetCount
//                     0.00%  1.3580us         2     679ns     329ns  1.0290us  cuDeviceGet
//                     0.00%     698ns         1     698ns     698ns     698ns  cuDeviceTotalMem
//                     0.00%     568ns         1     568ns     568ns     568ns  cuDeviceGetUuid

const int len = 32 * 1024  * 1024;

template <int BLOCKSIZE>
__global__ void reduce_naive_kernel(int* arr, int* out, int len) {
  __shared__ int sdata[BLOCKSIZE];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bdim = blockDim.x;

  // i是arr的索引
  int i = bid * bdim + tid;

  if (i < len) {
    sdata[tid] = arr[i];
  }

  __syncthreads();

  for (int s = 1; s < bdim; s *= 2) {
    // 这里有缺陷。线程会被分到多个warp上执行，每个warp上的线程需要执行相同的指令
    // 同一时间，warp里的线程要么在if，要么在else
    // 所以这里会有阻塞。Warp Divergent
    // 而且取模操作比较昂贵
    if (tid % (2 * s) == 0 && i + s < len) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[bid] = sdata[0];
  }
}


int main() {
  int* arr = new int[len];
  int* out = new int[len];

  int *d_arr, *d_out;

  for (int i = 0; i < len; ++i) {
    arr[i] = i;
  }

  cudaMalloc((void**)&d_arr, sizeof(int) * len);
  cudaMalloc((void**)&d_out, sizeof(int) * len);

  cudaMemcpy(d_arr, arr, sizeof(int) * len, cudaMemcpyHostToDevice);

  const int blocksize = 256;
  const int gridsize = (len + blocksize - 1) / blocksize;

  reduce_naive_kernel<blocksize><<<gridsize, blocksize>>>(d_arr, d_out, len);

  cudaMemcpy(out, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);

  int sum = 0;
  for (int i = 0; i < gridsize; ++i) {
    sum += out[i];
  }
  printf("sum = %d\n", sum);

  // 核对结果
  int sum2 = 0;
  for (int i = 0; i < len; i++) {
    sum2 += arr[i];
  }

  if (sum == sum2) {
    printf("success\n");
  } else {
    printf("failed\n");
  }

  // 释放内存
  cudaFree(d_arr);
  cudaFree(d_out);
  delete[] arr;
  delete[] out;
  return 0;
}