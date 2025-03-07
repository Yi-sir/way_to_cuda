#include <iostream>
#include <chrono>

// ==2123796== NVPROF is profiling process 2123796, command: ./naive_conv
// [blockDim.x, blockDim.y, gridDim.x, gridDim.y] = [64, 16, 1009, 1]
// Verification passed!
// ==2123796== Profiling application: ./naive_conv
// ==2123796== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   52.98%  12.879ms         1  12.879ms  12.879ms  12.879ms  naive_conv2d_kernel(int, int, int, int, int, int, int, int, int, int, int, int, int, float*, float*, float*)
//                    33.63%  8.1754ms         1  8.1754ms  8.1754ms  8.1754ms  [CUDA memcpy DtoH]
//                    13.39%  3.2538ms         3  1.0846ms     960ns  2.9464ms  [CUDA memcpy HtoD]
//       API calls:   86.49%  167.19ms         3  55.729ms  50.427us  167.08ms  cudaMalloc
//                     6.66%  12.881ms         1  12.881ms  12.881ms  12.881ms  cudaDeviceSynchronize
//                     6.55%  12.652ms         4  3.1630ms  89.028us  9.1152ms  cudaMemcpy
//                     0.15%  281.08us         3  93.693us  49.697us  175.99us  cudaFree
//                     0.12%  224.41us        97  2.3130us     289ns  86.248us  cuDeviceGetAttribute
//                     0.02%  38.671us         1  38.671us  38.671us  38.671us  cudaLaunchKernel
//                     0.01%  21.305us         1  21.305us  21.305us  21.305us  cuDeviceGetName
//                     0.01%  13.996us         1  13.996us  13.996us  13.996us  cuDeviceGetPCIBusId
//                     0.00%  3.0260us         3  1.0080us     270ns  2.3270us  cuDeviceGetCount
//                     0.00%  1.4290us         2     714ns     295ns  1.1340us  cuDeviceGet
//                     0.00%     599ns         1     599ns     599ns     599ns  cuDeviceTotalMem
//                     0.00%     544ns         1     544ns     544ns     544ns  cuDeviceGetUuid
//                     0.00%     368ns         1     368ns     368ns     368ns  cudaGetErrorString
//                     0.00%     248ns         1     248ns     248ns     248ns  cudaGetLastError

/*
 * @param n: batch size
 * @param c: 通道数
 * @param h: 输入数据高
 * @param w: 输入数据宽
 * @param k: 卷积核数量
 * @param r: 卷积核高
 * @param s: 卷积核宽
 * @param out_h: 输出数据高
 * @param out_w: 输出数据宽
 * @param u: 卷积在高方向上的步长
 * @param v: 卷积在宽方向上的步长
 * @param p: 卷积在高方向上的补边
 * @param q: 卷积在宽方向上的补边
 * @param in: 输入数据
 * @param weight: 卷积核
 * @param out: 输出数据
 */
__global__ void naive_conv2d_kernel(int n, int c, int h, int w, int k, int r,
                                    int s, int out_h, int out_w, int u, int v,
                                    int p, int q, float* in, float* weight,
                                    float* out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z;

  if (x >= out_h * out_w || y >= k || z >= n) {
    return;
  }

  int pos_out_h = x / out_w;
  int pos_out_w = x % out_w;

  float sum = 0.0;

  int pos_ori_h = pos_out_h * u - p;
  int pos_ori_w = pos_out_w * v - q;

  int in_offset = z * c * h * w + pos_ori_h * w + pos_ori_w;
  int weight_offset = y * c * r * s;
  int in_channel_offset = h * w;
  int weight_channel_offset = r * s;

  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < s; ++j) {
      int pos_real_h = pos_ori_h + i;
      int pos_real_w = pos_ori_w + j;
      if (pos_real_h < 0 || pos_real_w < 0 || pos_real_h >= h ||
          pos_real_w >= w) {
        continue;
      }
      for (int channel = 0; channel < c; ++channel) {
        sum +=
            in[in_offset + channel * in_channel_offset + i * w + j] *
            weight[weight_offset + channel * weight_channel_offset + i * s + j];
      }
    }
  }
  int out_offset = z * k * out_w * out_h + y * out_h * out_w + x;
  out[out_offset] = sum;
}

void naive_conv2d_cpu(int n, int c, int h, int w, int k, int r, int s,
                      int out_h, int out_w, int u, int v, int p, int q,
                      float* in, float* weight, float* out) {
  // Out(N_i, C_out_j) = sigma_k=0_k=C_in-1(weight(C_out_j, k) * input(N_i, k))
  // batch loop
  for (int n_i = 0; n_i < n; ++n_i) {
    int n_input_offset = n_i * c * h * w;
    // output channel loop

    for (int k_i = 0; k_i < k; ++k_i) {
      int k_offset = k_i * c * r * s;

      // output pixel loop
      for (int out_i = 0; out_i < out_h; ++out_i) {
        for (int out_j = 0; out_j < out_w; ++out_j) {
          // 输出[out_i, out_j]对应输入的某个区域的起始点
          int input_start_i = out_i * u - p;
          int input_start_j = out_j * v - q;
          double sum = 0.0;

          for (int c_i = 0; c_i < c; ++c_i) {
            int c_kernel_offset = c_i * r * s;
            int c_input_offset = c_i * h * w;

            for (int weight_i = 0; weight_i < r; ++weight_i) {
              for (int weight_j = 0; weight_j < s; ++weight_j) {
                // 权重的位置
                int weight_offset =
                    k_offset + c_kernel_offset + weight_i * s + weight_j;
                // 原图区域里当前权重对应点的坐标
                int input_current_i = input_start_i + weight_i;
                int input_current_j = input_start_j + weight_j;

                if (input_current_i >= 0 && input_current_j >= 0 &&
                    input_current_i < h && input_current_j < w) {
                  // 原图数据的位置
                  int input_offset = n_input_offset + c_input_offset +
                                     input_current_i * w + input_current_j;
                  sum += (double)in[input_offset] * weight[weight_offset];
                }
              }
            }
          }
          int output_offset = n_i * k * out_h * out_w + k_i * out_h * out_w +
                              out_i * out_w + out_j;
          out[output_offset] = (float)sum;
        }
      }
    }
  }
}

void conv2d_cpu(float* in, float* pwei, float* out, int n, int c, int h, int w,
                int k, int r, int s, int u, int v, int p, int q) {
  int out_h = (h + 2 * p - r) / u + 1;
  int out_w = (w + 2 * q - s) / v + 1;

  for (int n_num = 0; n_num < n; n_num++) {
    for (int k_num = 0; k_num < k; k_num++) {
      for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
          double sum = 0.0;
          int pos_h = i * u - p;
          int pos_w = j * v - q;

          for (int c_num = 0; c_num < c; c_num++) {
            for (int kh_num = 0; kh_num < r; kh_num++) {
              for (int kwNum = 0; kwNum < s; kwNum++) {
                int pos_ori_h = pos_h + kh_num;
                int pos_ori_w = pos_w + kwNum;
                if (pos_ori_w >= 0 && pos_ori_h >= 0 && pos_ori_w < w &&
                    pos_ori_h < h) {
                  sum += (double)(in[n_num * c * h * w + c_num * (w * h) +
                                     pos_ori_h * w + pos_ori_w] *
                                  pwei[k_num * r * s * c + c_num * r * s +
                                       kh_num * s + kwNum]);
                }
              }
            }
          }

          out[n_num * k * out_h * out_w + k_num * out_h * out_w + i * out_w +
              j] = (float)sum;
        }
      }
    }
  }
}

int main() {
  // 定义输入数据和卷积核的尺寸
  const int n = 4;                            // batch size
  const int c = 3;                            // 通道数
  const int h = 256;                          // 数据高
  const int w = 256;                          // 数据宽
  const int k = 16;                           // 卷积核数量
  const int r = 3;                            // 卷积核高
  const int s = 3;                            // 卷积核宽
  const int u = 1;                            // 卷积在高方向上的步长
  const int v = 1;                            // 卷积在宽方向上的步长
  const int p = 0;                            // 卷积在高方向上的补边
  const int q = 0;                            // 卷积在宽方向上的补边
  const int out_h = (h - r + 2 * p) / u + 1;  // 输出高
  const int out_w = (w - s + 2 * q) / v + 1;  // 输出宽

  float *in_device, *weight_device, *out_device;

  float* in = new float[n * c * h * w];
  float* weight = new float[k * c * r * s];
  float* out = new float[n * k * out_h * out_w];
  float* out_cpu = new float[n * k * out_h * out_w];

  cudaMalloc(&in_device, sizeof(float) * n * c * h * w);
  cudaMalloc(&weight_device, sizeof(float) * k * c * r * s);
  cudaMalloc(&out_device, sizeof(float) * n * k * out_h * out_w);

  for (int i = 0; i < n * c * h * w; ++i) {
    in[i] = (float)rand() / RAND_MAX;
  }
  for (int i = 0; i < k * c * r * s; ++i) {
    weight[i] = (float)rand() / RAND_MAX;
  }

  cudaMemcpy(in_device, in, sizeof(float) * n * c * h * w,
             cudaMemcpyHostToDevice);
  cudaMemcpy(weight_device, weight, sizeof(float) * k * c * r * s,
             cudaMemcpyHostToDevice);
  cudaMemcpy(out_device, out, sizeof(float) * n * k * out_h * out_w,
             cudaMemcpyHostToDevice);

  const int blockDim_x =
      (out_h * out_w / k) > 1024/k ? 1024/k : (out_h * out_w / k);
  const int blockDim_y = k;

  const int gridDim_x = (out_h * out_w + blockDim_x - 1) / blockDim_x;
  const int gridDim_y = (k + blockDim_y - 1) / blockDim_y;

  std::cout << "[blockDim.x, blockDim.y, gridDim.x, gridDim.y] = ["
            << blockDim_x << ", " << blockDim_y << ", " << gridDim_x << ", "
            << gridDim_y << "]\n";

  dim3 blockDim{blockDim_x, blockDim_y};
  dim3 gridDim{gridDim_x, gridDim_y, n};

  naive_conv2d_kernel<<<gridDim, blockDim>>>(n, c, h, w, k, r, s, out_h, out_w,
                                             u, v, p, q, in_device,
                                             weight_device, out_device);
  cudaDeviceSynchronize();

  auto cuda_err = cudaGetLastError();
  auto cuda_err_str = cudaGetErrorString(cuda_err);

  if (cuda_err != cudaSuccess) {
    std::cout << cuda_err_str << std::endl;
  }

  cudaMemcpy(out, out_device, sizeof(float) * n * k * out_h * out_w,
             cudaMemcpyDeviceToHost);

  auto start_time = std::chrono::high_resolution_clock::now();
  naive_conv2d_cpu(n, c, h, w, k, r, s, out_h, out_w, u, v, p, q, in, weight,
                   out_cpu);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  std::cout << "naive_conv2d_cpu costs: " << duration << "ms.\n";
  
  bool pass = true;
  for (int i = 0; i < n * k * out_h * out_w; ++i) {
    if (abs(out[i] - out_cpu[i]) > 1e-5) {
      pass = false;
      std::cout << "Verification failed at " << i << "!" << std::endl;
      std::cout << "CPU: " << out_cpu[i] << " GPU: " << out[i] << std::endl;
      break;
    }
  }
  if (pass) {
    std::cout << "Verification passed!" << std::endl;
  }

  delete in;
  delete weight;
  delete out;
  delete out_cpu;

  cudaFree(in_device);
  cudaFree(weight_device);
  cudaFree(out_device);

  return 0;
}