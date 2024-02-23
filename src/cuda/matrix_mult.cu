#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

// With this implementation and 256 threads per block, works only for up to 16M. Why?
const int N = 15 * 1024 * 1024;
const dim3 BLOCK_DIM = 256;

// Simple CPU implementation of a single precision AXPY operation.
void CpuSaxpy(int n, float alpha, const float* x, float* y) {
  for (int i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

// GPU implementation of AXPY operation - one CUDA thread per vector element.
__global__ void GpuSaxpy(int n, float alpha, const float* x, float* y) {
  // Calculate index of the vector element updated by this thread.
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Some threads might be surplus. They should not perform any operations,
  // as it may result in accessing unallocated memory or other unexpected behaviour.
  if (idx < n) {
    y[idx] += alpha * x[idx];
  }
}

// GPU implementation of AXPY operation - CUDA thread updates multiple vector elements.
__global__ void GpuSaxpyMulti(int n, float alpha, const float* x, float* y) {
  // Calculate index of the first vector element updated by this thread.
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  // The total number of CUDA threads.
  int grid_size = blockDim.x * gridDim.x;

  for (; idx < n; idx += grid_size) {
    y[idx] += alpha * x[idx];
  }
}

int main(int argc, char** argv) {

// ----------------------- Host memory initialisation ----------------------- //

  float* h_x = new float[N];
  float* h_y = new float[N];

  // Initialise vectors on the CPU.
  std::fill_n(h_x, N, 1.0f);  // Vector of ones
  for (int i = 0; i < N; ++i) {
    h_y[i] = 0.33f * (i + 1);
  }

// ---------------------- Device memory initialisation ---------------------- //

  // Allocate global memory on the GPU.
  float* d_x = 0;
  checkCudaErrors(cudaMalloc((void**) &d_x, N * sizeof(float)));

  // There's also a template version of cudaMalloc function, avoiding (void**) cast.
  float* d_y = 0;
  checkCudaErrors(cudaMalloc<float>(&d_y, N * sizeof(float)));

  // Copy vectors from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
  // cudaMemcpy() is blocking, i.e. CPU won't proceed until all data is copied.

// --------------------- Calculations for CPU implementation ---------------- //

  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  timer->start();
  CpuSaxpy(N, 0.25f, h_x, h_y);  // y = y + 0.25 * x;
  CpuSaxpy(N, -10.5f, h_x, h_y);  // y = y - 10.5 * x;

  timer->stop();
  std::cout << "CPU time: " << timer->getTime() << " ms." << std::endl;

// --------------------- Calculations for GPU implementation ---------------- //

  // Calculate the number of required thread blocks.
  const dim3 GRID_DIM = (N - 1) / BLOCK_DIM.x + 1;

  timer->reset();
  timer->start();
  //GpuSaxpy<<<GRID_DIM, BLOCK_DIM>>>(N, 0.25f, d_x, d_y);
  //GpuSaxpy<<<GRID_DIM, BLOCK_DIM>>>(N, -10.5f, d_x, d_y);

  // This should work as well.
  GpuSaxpyMulti<<<GRID_DIM, BLOCK_DIM>>>(N, 0.25f, d_x, d_y);
  GpuSaxpyMulti<<<GRID_DIM, BLOCK_DIM>>>(N, -10.5f, d_x, d_y);

  // Kernel calls are asynchronous with respect to the host, i.e. control is returned to
  // the CPU immediately. It is possible that the second operation is submitted _before_
  // the first one is completed. However, CUDA driver will ensure that they will be
  // completed in FIFO order, one at a time.

  // CPU has to explicitly wait for the device to complete
  // in order to get meaningful time measurement.
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();
  std::cout << "GPU time: " << timer->getTime() << " ms." << std::endl;

  // Download the resulting vector y from the device and store it in h_x.
  checkCudaErrors(cudaMemcpy(h_x, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
  // cudaMemcpy is synchronous, i.e. it will wait for any computation on the GPU to
  // complete before any data is copied (as if cudaDeviceSynchronize() was called before).

  // Now let's check if the results are the same.
  float diff = 0.0f;
  for (int i = 0; i < N; ++i) {
    diff = std::max(diff, std::abs(h_x[i] - h_y[i]));
  }
  std::cout << "Max diff = " << diff << std::endl;

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  // Don't forget to free host and device memory!
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

  delete[] h_x;
  delete[] h_y;

  return 0;
}
