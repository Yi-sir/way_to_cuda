#include <cuda_runtime.h>

#include <iostream>

// ==1960676== NVPROF is profiling process 1960676, command: ./reduce_interleaved
// sum = -16777216
// success
// ==1960676== Profiling application: ./reduce_interleaved
// ==1960676== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   54.09%  66.633ms         1  66.633ms  66.633ms  66.633ms  [CUDA memcpy DtoH]
//                    30.96%  38.143ms         1  38.143ms  38.143ms  38.143ms  void reduce_naive_kernel<int=256>(int*, int*, int)
//                    14.94%  18.409ms         1  18.409ms  18.409ms  18.409ms  [CUDA memcpy HtoD]
//       API calls:   65.02%  232.20ms         2  116.10ms  61.842us  232.14ms  cudaMalloc
//                    34.81%  124.32ms         2  62.160ms  18.506ms  105.81ms  cudaMemcpy
//                     0.08%  294.29us         2  147.15us  99.380us  194.91us  cudaFree
//                     0.07%  235.39us        97  2.4260us     331ns  91.535us  cuDeviceGetAttribute
//                     0.01%  33.105us         1  33.105us  33.105us  33.105us  cudaLaunchKernel
//                     0.01%  22.446us         1  22.446us  22.446us  22.446us  cuDeviceGetName
//                     0.00%  11.209us         1  11.209us  11.209us  11.209us  cuDeviceGetPCIBusId
//                     0.00%  3.7810us         3  1.2600us     343ns  2.9900us  cuDeviceGetCount
//                     0.00%  1.3080us         2     654ns     328ns     980ns  cuDeviceGet
//                     0.00%     676ns         1     676ns     676ns     676ns  cuDeviceTotalMem
//                     0.00%     527ns         1     527ns     527ns     527ns  cuDeviceGetUuid

// const int len = 32 * 1024  * 1024;
const int len = 1111111;

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

  // 交错寻址，目标是让尽可能多的线程参与运算
  for (int s = 1; s < bdim; s *= 2) {
    // 根据tid计算出sdata上当前需要收集数据的索引
    int index = 2 * s * tid;
    // s是被reduce的间距，index + s不能超过bdim，即thread num
    // bid * bdim + index + s < len，原本是bid*bdim+s<len，修改之后是对的
    // 这样的好处是，原本只有偶数线程在运算，现在奇数线程也可以参与运算了
    // 从而使得一个warp里尽可能多的线程都在做相同的事情
    if ((index + s < bdim) && (bid * bdim + index + s < len)) {
      sdata[index] += sdata[index + s];
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