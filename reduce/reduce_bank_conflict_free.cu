#include <cuda_runtime.h>

#include <iostream>

// ==1966732== NVPROF is profiling process 1966732, command: ./reduce_bank
// sum = -16777216
// success
// ==1966732== Profiling application: ./reduce_bank
// ==1966732== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   57.80%  68.020ms         1  68.020ms  68.020ms  68.020ms  [CUDA memcpy DtoH]
//                    25.11%  29.544ms         1  29.544ms  29.544ms  29.544ms  void reduce_naive_kernel<int=256>(int*, int*, int)
//                    17.09%  20.112ms         1  20.112ms  20.112ms  20.112ms  [CUDA memcpy HtoD]
//       API calls:   64.47%  216.77ms         2  108.38ms  66.373us  216.70ms  cudaMalloc
//                    35.34%  118.81ms         2  59.407ms  20.211ms  98.603ms  cudaMemcpy
//                     0.09%  305.62us         2  152.81us  96.697us  208.93us  cudaFree
//                     0.07%  249.04us        97  2.5670us     327ns  95.236us  cuDeviceGetAttribute
//                     0.01%  44.438us         1  44.438us  44.438us  44.438us  cudaLaunchKernel
//                     0.01%  26.143us         1  26.143us  26.143us  26.143us  cuDeviceGetName
//                     0.01%  17.537us         1  17.537us  17.537us  17.537us  cuDeviceGetPCIBusId
//                     0.00%  3.7560us         3  1.2520us     313ns  2.9080us  cuDeviceGetCount
//                     0.00%  1.3020us         2     651ns     306ns     996ns  cuDeviceGet
//                     0.00%     593ns         1     593ns     593ns     593ns  cuDeviceTotalMem
//                     0.00%     571ns         1     571ns     571ns     571ns  cuDeviceGetUuid

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

  // blockDim.x = 256
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