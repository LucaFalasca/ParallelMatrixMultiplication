// 
// Author: Salvatore Filippone salvatore.filippone@cranfield.ac.uk
//

// Computes matrix-vector product. Matrix A is in row-major order
// i.e. A[i, j] is stored in i * k + j element of the vector.
//

#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

//#define _DEBUG
// Simple 1-D thread block
// Size should be at least 1 warp 
#define BD 2

const dim3 BLOCK_DIM(BD);

// Simple CPU implementation of matrix addition.
void CpuMatrixVector(int rows, int cols, const float* A, const float* x, float* y) {
  for (int row = 0; row < rows; ++row) {
    float t=0.0;
    for (int col = 0; col < cols; ++col) {
      int idx = row * cols + col;
      t += A[idx] * x[col];
    }
    y[row] = t;
  }
}

// GPU implementation of matrix_vector product using a block of threads for
// each row. 
__device__ void rowReduce(volatile float *sdata, int tid) {
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void rowReduce2(volatile float *sdata, int tid, int s) {
  switch(s){
  case 16:  sdata[tid] += sdata[tid + 16];
  case  8:  sdata[tid] += sdata[tid +  8];
  case  4:  sdata[tid] += sdata[tid +  4];
  case  2:  sdata[tid] += sdata[tid +  2];
  case  1:  sdata[tid] += sdata[tid +  1];
  }
}

__global__ void gpuMatrixVector(int rows, int cols, int n, const float* A,
				const float* x, float* y) {
  __shared__ float aux[BD];
  int tc     = threadIdx.x;
  int row    = blockIdx.x;
  if (row < rows) {
    // Starting address of indexing within matrix A
    for(int i = 0; i < n; i++){
      int idxm = row*cols+tc;
      float t  = 0.0;
      int q = 0;
      aux[tc] = 0.0;
      for (int ic= tc;  ic<cols; ic += blockDim.x) {
        int icx = i * n + ic;
        t += A[idxm]*x[icx];
        #ifdef _DEBUG
        printf("{Blocco %d} P%d-A[%d]: %f --- x[%d]: %f\n", row, tc, idxm, A[idxm], icx, x[icx]); 
        #endif
        q++;
        idxm +=  blockDim.x;
      }
      aux[tc] = t;
    
    
    __syncthreads();
      for (int s=BD/2; s >=32; s >>=1){
        if (tc<s)
          aux[tc] += aux[tc+s]; 
        __syncthreads();
      }
    
      
      if (tc<16) rowReduce(aux,tc);
      
      if (tc == 0)
        y[i + row * n] = aux[tc];
    }
  }
}

int main(int argc, char** argv) {

  if (argc < 3) {
    fprintf(stderr,"Usage: %s  rows cols\n",argv[0]);
  }
  int m=atoi(argv[1]);
  int k=atoi(argv[2]);
  int n=atoi(argv[3]);
  
  
  // ----------------------- Host memory initialisation ----------------------- //

  float* h_A = new float[m * k];
  float* h_x = new float[k * n];
  float* h_y = new float[m * n];
  float* h_y_d = new float[m * n];

  srand(123456);
  #ifdef _DEBUG 
  std::cout << "Matrix A: " << std::endl;
  #endif
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < k; ++col) {
      int idx = row * k + col;
      h_A[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
      #ifdef _DEBUG 
      std::cout << "|" << h_A[idx] << "";
      #endif
    }
    #ifdef _DEBUG 
    std::cout << "|" << std::endl;
    #endif
    h_y[row] = 0.0;
  }
  #ifdef _DEBUG 
  std::cout << "\nMatrix x:" << std::endl;
  #endif
  for (int col = 0; col < k; ++col) {
    for(int row = 0; row < n; ++row){
      int idx = row * k + col;
      h_x[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
      #ifdef _DEBUG 
      std::cout << "|" << h_x[idx] << "";
      #endif
    }
    #ifdef _DEBUG 
    std::cout << "|"<< std::endl;
    #endif
  }
  

  std::cout << "Matrix-vector product: 1D thread block version " << std::endl;
  std::cout << "Test case: " << m  << " x " << k << std::endl;
// ---------------------- Device memory initialisation ---------------------- //

  float *d_A, *d_x, *d_y;

  checkCudaErrors(cudaMalloc((void**) &d_A, m * k * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_x, k * n * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_y, m * n * sizeof(float)));

  // Copy matrices from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_x, h_x, k * n * sizeof(float), cudaMemcpyHostToDevice));

  // ------------------------ Calculations on the CPU ------------------------- //
  
  float flopcnt=2.e-6*m*k*n;
  
  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);
  /*
  timer->start();
  CpuMatrixVector(m, k, h_A, h_x, h_y);

  timer->stop();
  float cpuflops=flopcnt/ timer->getTime();
  std::cout << "  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;
  */

// ------------------------ Calculations on the GPU ------------------------- //

  // Calculate the dimension of the grid of blocks (1D) needed to cover all
  // entries in the matrix and output vector
  const dim3 GRID_DIM(m,1);

  timer->reset();
  timer->start();
  gpuMatrixVector<<<GRID_DIM, BLOCK_DIM >>>(m, k, n, d_A, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  float gpuflops=flopcnt/ timer->getTime();
  std::cout << "  GPU time: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

  // Download the resulting vector d_y from the device and store it in h_y_d.
  checkCudaErrors(cudaMemcpy(h_y_d, d_y, m*n*sizeof(float),cudaMemcpyDeviceToHost));

  //print h_y_d
  #ifdef _DEBUG
  std::cout << "Matrix-vector product: 1D thread block version " << std::endl;
  std::cout << "Test case: " << m  << " x " << n << std::endl;
  std::cout << "Matrix y_d: " << std::endl;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int idx = row * n + col;
      std::cout << "|" << h_y_d[idx] << "";
    }
    std::cout << "|" << std::endl;
  }
  #endif

  // Now let's check if the results are the same.
  float reldiff = 0.0f;
  float diff = 0.0f;
  
  for (int row = 0; row < m; ++row) {
    float maxabs = std::max(std::abs(h_y[row]),std::abs(h_y_d[row]));
    if (maxabs == 0.0) maxabs=1.0;
    reldiff = std::max(reldiff, std::abs(h_y[row] - h_y_d[row])/maxabs);
    diff = std::max(diff, std::abs(h_y[row] - h_y_d[row]));
    //std::cout << row<<" "<<h_y[row]<<" "<<h_y_d[row] <<std::endl;
  }
  std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;
  // Rel diff should be as close as possible to unit roundoff; float
  // corresponds to IEEE single precision, so unit roundoff is
  // 1.19e-07
  // 

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));
  
  delete[] h_A;
  delete[] h_x;
  delete[] h_y;
  delete[] h_y_d;
  
  return 0;
}
