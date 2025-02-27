#include <cuda_runtime.h>

#include <iostream>

// ==1967764== NVPROF is profiling process 1967764, command: ./reduce_idle
// sum = -16777216
// success
// ==1967764== Profiling application: ./reduce_idle
// ==1967764== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   74.32%  97.965ms         1  97.965ms  97.965ms  97.965ms  [CUDA memcpy DtoH]
//                    14.05%  18.527ms         1  18.527ms  18.527ms  18.527ms  [CUDA memcpy HtoD]
//                    11.63%  15.328ms         1  15.328ms  15.328ms  15.328ms  void reduce_naive_kernel<int=256>(int*, int*, int)
//       API calls:   58.93%  192.17ms         2  96.086ms  80.827us  192.09ms  cudaMalloc
//                    40.87%  133.29ms         2  66.646ms  18.618ms  114.68ms  cudaMemcpy
//                     0.10%  329.98us         2  164.99us  108.71us  221.28us  cudaFree
//                     0.07%  241.72us        97  2.4910us     331ns  91.829us  cuDeviceGetAttribute
//                     0.01%  44.596us         1  44.596us  44.596us  44.596us  cudaLaunchKernel
//                     0.01%  23.277us         1  23.277us  23.277us  23.277us  cuDeviceGetName
//                     0.00%  9.3690us         1  9.3690us  9.3690us  9.3690us  cuDeviceGetPCIBusId
//                     0.00%  4.3700us         3  1.4560us     395ns  3.2150us  cuDeviceGetCount
//                     0.00%  2.1670us         2  1.0830us     346ns  1.8210us  cuDeviceGet
//                     0.00%     770ns         1     770ns     770ns     770ns  cuDeviceTotalMem
//                     0.00%     602ns         1     602ns     602ns     602ns  cuDeviceGetUuid

// T400 peak bandwidth is 80GB/s, in this case, bandwidth is 
// 4 * 32 * 1024 * 1024 Bytes / (15.328 * 10-3 s) = 8.35 GB/s

const int len = 32 * 1024 * 1024;

template <int BLOCKSIZE>
__global__ void reduce_naive_kernel(int* arr, int* out, int len) {
  __shared__ int sdata[BLOCKSIZE];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int bdim = blockDim.x;

  // i是arr的索引，这里一个线程在拷贝内存到shared
  // mem的时候要顺便做一个加法，所以线程跨度大了
  int i = bid * bdim * 2 + tid;

  if (i < len) {
    sdata[tid] = arr[i] + arr[i + bdim];
  }

  __syncthreads();

  // blockDim.x = 256
  // 第一次循环，0-127号线程去128-255位置取数据求和
  // 考虑到bank的排列，每128字节存放在32个bank的一层里，而128个int共占用4层
  // 所以，0号线程使用的是bank0的第0层和第4层
  // 1号线程使用的是bank1的第0层和第4层
  // 32个线程是一个warp，第二个warp的第0号线程使用的是bank0的第1层和第5层
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
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
  // 注意这里因为每个线程干的活变多了，所以block数量变少了
  const int gridsize = (len + blocksize - 1) / (blocksize * 2);

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