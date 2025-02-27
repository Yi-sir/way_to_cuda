#include <cuda_runtime.h>

#include <iostream>

// ==1969270== NVPROF is profiling process 1969270, command: ./reduce_unroll
// sum = -16777216
// success
// ==1969270== Profiling application: ./reduce_unroll
// ==1969270== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   70.43%  66.969ms         1  66.969ms  66.969ms  66.969ms  [CUDA memcpy DtoH]
//                    19.54%  18.582ms         1  18.582ms  18.582ms  18.582ms  [CUDA memcpy HtoD]
//                    10.03%  9.5332ms         1  9.5332ms  9.5332ms  9.5332ms  void reduce_naive_kernel<int=256>(int*, int*, int)
//       API calls:   68.61%  211.75ms         2  105.88ms  74.887us  211.68ms  cudaMalloc
//                    31.18%  96.212ms         2  48.106ms  18.679ms  77.533ms  cudaMemcpy
//                     0.10%  309.81us         2  154.91us  102.43us  207.39us  cudaFree
//                     0.08%  259.47us        97  2.6740us     316ns  93.137us  cuDeviceGetAttribute
//                     0.01%  42.997us         1  42.997us  42.997us  42.997us  cudaLaunchKernel
//                     0.01%  23.829us         1  23.829us  23.829us  23.829us  cuDeviceGetName
//                     0.00%  10.226us         1  10.226us  10.226us  10.226us  cuDeviceGetPCIBusId
//                     0.00%  3.4370us         3  1.1450us     282ns  2.6170us  cuDeviceGetCount
//                     0.00%  1.6310us         2     815ns     350ns  1.2810us  cuDeviceGet
//                     0.00%  1.2080us         1  1.2080us  1.2080us  1.2080us  cuDeviceTotalMem
//                     0.00%     521ns         1     521ns     521ns     521ns  cuDeviceGetUuid

const int len = 32 * 1024 * 1024;

__device__ void warp_reduce(volatile int* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

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
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    warp_reduce(sdata, tid);
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