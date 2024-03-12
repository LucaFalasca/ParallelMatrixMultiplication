// 
// Author: Luca Falasca
//

// Computes matrix-matrix product.
//

#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include <cuda_profiler_api.h>
#include <math.h>

#define _DEBUG
// Simple 1-D thread block
// Size should be at least 1 warp 
#define BD 2
#define BD2 24
#define COLS 2

const dim3 BLOCK_DIM(BD);

// Simple CPU implementation of matrix addition.
void CpuMatrixVector(int m, int k, int n, const float* A, const float* x, float* y) {
  float t=0.0;
  for (int row = 0; row < m; ++row) {
    for(int i = 0; i < n; i++){
      t = 0.0;
      for (int col = 0; col < k; ++col) {
        int idx = row * k + col;
        int icx = i * k + col;
        t += A[idx] * x[icx];
        #ifdef _DEBUG
        printf("CPU - A[%d]: %f --- B[%d]: %f\n", idx, A[idx], icx, x[icx]); 
        #endif
      }
      y[i + row * n] += t;
    }
  }
}

// GPU implementation of matrix_vector product using a block of threads for
// each row. 
__device__ void rowReduce(volatile float *sdata, int tid) {
  #ifdef _DEBUG
  if(blockIdx.x == 0){
    printf("sdata[%d]: %f\n", tid, sdata[tid]);
    printf("sdata[%d + 16]: %f\n", tid, sdata[tid + 16]);
    printf("sdata[%d + 8]: %f\n", tid, sdata[tid + 8]);
    printf("sdata[%d + 4]: %f\n", tid, sdata[tid + 4]);
    printf("sdata[%d + 2]: %f\n", tid, sdata[tid + 2]);
    printf("sdata[%d + 1]: %f\n", tid, sdata[tid + 1]);
  }
  #endif
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void rowReduce2(volatile float *sdata, int tid, int s) {
  #ifdef _DEBUG
  
  if(blockIdx.x == 0){
    printf("row->%d sdata[%d]: %f\n", blockIdx.x, tid, sdata[tid]);
    printf("row->%d sdata[%d + 16]: %f\n", blockIdx.x, tid, sdata[tid + 16]);
    printf("row->%d sdata[%d + 8]: %f\n", blockIdx.x, tid, sdata[tid + 8]);
    printf("row->%d sdata[%d + 4]: %f\n", blockIdx.x, tid, sdata[tid + 4]);
    printf("row->%d sdata[%d + 2]: %f\n", blockIdx.x, tid, sdata[tid + 2]);
    printf("row->%d sdata[%d + 1]: %f\n", blockIdx.x, tid, sdata[tid + 1]);
  }
  
  #endif
  switch(s){
  case 16:  sdata[tid] += sdata[tid + 16];
  case  8:  sdata[tid] += sdata[tid +  8];
  case  4:  sdata[tid] += sdata[tid +  4];
  case  2:  sdata[tid] += sdata[tid +  2];
  case  1:  sdata[tid] += sdata[tid +  1];
  }
}

__global__ void gpuMatrixMatrix(int m, int k, int n, const float* A,
				const float* B, float* C) {
  __shared__ float aux[COLS][BD];
  int tc     = threadIdx.x;
  int row    = blockIdx.x;
  int s = min(16,BD/2);
  int n_cols = COLS;
  __shared__ float a_shared[BD];
  
  if (row < m) {
    // Itero per ogni blocco di colonne della matrice B
    for(int i = 0; i < n; i+= COLS){
      // Se l'ultimo blocco di colonne è minore di COLS aggiorno il numero di colonne del blocco
      if(i + COLS > n) n_cols = n % COLS; 

      // Inizializzo la matrice ausiliaria per la reduce
      for(int j = 0; j < n_cols; j++) aux[j][tc] = 0.0;


      /*
        Ogni processo si calcola il prodotto tra il valore della riga di A e gli elementi corrispondenti 
        delle colonne del blocco di colonne di B 
      */
      int idxm = row*k+tc;  //indice dell'elemento della riga A associato al processo

      // Itero sui blocchi della riga di A 
      for (int ic= tc;  ic<k; ic += blockDim.x) {
        //Ogni processo ha in memoria condivisa il valore del'elemento della riga di A corrispondente
        a_shared[tc] = A[idxm];
        // Itero sulle colonne del blocco di colonne di B
        for(int j = 0; j < n_cols; j++){
          // Calcolo l'indice dell'elemento di B corrispondente
          int icx = (i + j) * k + ic;
          aux[j][tc] += a_shared[tc]*B[icx];
          #ifdef _DEBUG
          printf("{Blocco %d} P%d-A[%d]: %f --- B[%d]: %f\n", row, tc, idxm, A[idxm], icx, B[icx]); 
          #endif
        }
        idxm +=  blockDim.x;
      }
      __syncthreads();

      //print aux matrix
      #ifdef _DEBUG
      if(tc == 0){
        for(int j = 0; j < n_cols; j++){
          for(int k = 0; k < BD; k++){
            printf("[BEFORE]row->%d aux[%d][%d]: %f\n", row, j, k, aux[j][k]);
          }
        }
      }
      #endif

      // Reduce
      for(int j = 0; j < n_cols; j++){
        for (int s2=BD/2; s2 >=32; s2 >>=1){
          if (tc<s2)
            aux[j][tc] += aux[j][tc+s2]; 
          __syncthreads();
        }
      }

      for(int j = 0; j < n_cols; j++){
        if (tc < s) rowReduce2(&(aux[j][0]),tc,s);

        if (tc == 0){
          #ifdef _DEBUG
          if(1)
          printf("[AFTER REDUCE]row->%d aux[%d][%d]: %f\n", row, j, tc, aux[j][tc]);
          printf("[FINAL] y[%d] = aux[%d][%d]: %f\n", i + j + row * n, j, tc, aux[j][tc]);
          #endif

          C[i + j + row * n] += aux[j][tc];
        }
      }
    
    }
  }
}

__global__ void gpuMatrixVectorV4(int m, int k, int n, const float* A,
				const float* B, float* y) {
  __shared__ float aux[BD2][BD];
  int tc     = threadIdx.x;
  int row    = blockIdx.x;
  int s = min(16,BD/2);
  int size = BD2;
  __shared__ float a_shared[BD];
  
  if (row < m) {
    
    // Itero per ogni colonna della matrice B (n)
    for(int i = 0; i < n; i+= BD2){
      if(i + BD2 > n) size = n % BD2;
      for(int j = 0; j < size; j++) aux[j][tc] = 0.0;
      int idxm = row*k+tc;
      for (int ic= tc;  ic<k; ic += blockDim.x) {
        a_shared[tc] = A[idxm];
        for(int j = 0; j < size; j++){
          int icx = i * k + ic + j * k;
          aux[j][tc] += a_shared[tc]*B[icx];
          #ifdef _DEBUG
          printf("{Blocco %d} P%d-A[%d]: %f --- B[%d]: %f\n", row, tc, idxm, A[idxm], icx, B[icx]); 
          #endif
        }
        idxm +=  blockDim.x;
      }
      
      
    
      __syncthreads();
      //print aux matrix
      #ifdef _DEBUG
      if(tc == 0){
        for(int j = 0; j < size; j++){
          for(int k = 0; k < BD; k++){
            printf("[BEFORE]row->%d aux[%d][%d]: %f\n", row, j, k, aux[j][k]);
          }
        }
      }
      #endif
      for(int j = 0; j < size; j++){
        for (int s2=BD/2; s2 >=32; s2 >>=1){
          if (tc<s2)
            aux[j][tc] += aux[j][tc+s2]; 
          __syncthreads();
        }
      }

      for(int j = 0; j < size; j++){
        if (tc < s) rowReduce2(&(aux[j][0]),tc,s);

        if (tc == 0){
          #ifdef _DEBUG
          if(1)
          printf("[AFTER REDUCE]row->%d aux[%d][%d]: %f\n", row, j, tc, aux[j][tc]);
          printf("[FINAL] y[%d] = aux[%d][%d]: %f\n", i + j + row * n, j, tc, aux[j][tc]);
          #endif

          y[i + j + row * n] += aux[j][tc];
        }
      }
    
    }
  }
}

__global__ void gpuMatrixVectorV3(int m, int k, int n, const float* A,
				const float* B, float* y) {
  __shared__ float aux[BD][BD2];
  int tc     = threadIdx.x;
  int row    = blockIdx.x;
  int s = min(16,BD/2);
  int size = BD2;
  
  if (row < m) {
    
    // Itero per ogni colonna della matrice B (n)
    for(int i = 0; i < n; i+= BD2){
      if(i + BD2 > n) size = n % BD2;
      for(int j = 0; j < size; j++){
        int idxm = row*k+tc;
        float t  = 0.0;
        int q = 0;
        aux[j][tc] = 0.0;
        for (int ic= tc;  ic<k; ic += blockDim.x) {
          int icx = i * k + ic + j * k;
          t += A[idxm]*B[icx];
          #ifdef _DEBUG
          printf("{Blocco %d} P%d-A[%d]: %f --- B[%d]: %f\n", row, tc, idxm, A[idxm], icx, B[icx]); 
          #endif
          q++;
          idxm +=  blockDim.x;
        }
        aux[j][tc] = t;
      }
      
      
    
      __syncthreads();
      //print aux matrix
      #ifdef _DEBUG
      if(tc == 0){
        for(int j = 0; j < size; j++){
          for(int k = 0; k < BD; k++){
            printf("[BEFORE]row->%d aux[%d][%d]: %f\n", row, j, k, aux[j][k]);
          }
        }
      }
      #endif
      for(int j = 0; j < size; j++){
        for (int s2=BD/2; s2 >=32; s2 >>=1){
          if (tc<s2)
            aux[j][tc] += aux[j][tc+s2]; 
          __syncthreads();
        }
      }

      for(int j = 0; j < size; j++){
        if (tc < s) rowReduce2(&(aux[j][0]),tc,s);

        if (tc == 0){
          #ifdef _DEBUG
          if(1)
          printf("[AFTER REDUCE]row->%d aux[%d][%d]: %f\n", row, j, tc, aux[j][tc]);
          printf("[FINAL] y[%d] = aux[%d][%d]: %f\n", i + j + row * n, j, tc, aux[j][tc]);
          #endif

          y[i + j + row * n] = aux[j][tc];
        }
      }
    
    }
  }
}


__global__ void gpuMatrixVectorV2(int m, int k, int n, const float* A,
				const float* B, float* y) {
  __shared__ float aux[BD];
  extern __shared__ float row_shared[];

  int tc     = threadIdx.x;
  int row    = blockIdx.x;
  int size_shared_vec = abs(k / BD);
  
  if (row < m) {
    
    // Itero per ogni colonna della matrice B (n)
    for(int i = 0; i < n; i++){
      int idxm = row*k+tc;
      float t  = 0.0;
      int q = 0;
      aux[tc] = 0.0;
      int irs = 0;
      //ogni processo prende e moltiplica solo alcuni pezzi della matrice A per i corrispendenti della colonna i di B
      for (int ic= tc;  ic<k; ic += blockDim.x) {
        int icx = i * n + ic;
        irs = q + size_shared_vec * tc + k % BD * (tc != 0);
        if(i == 0){
          //Se sto facendo la riga per la prima colonna allora prendo i valori dalla memoria globale e li metto in shared
          //dato che mi serviranno per le altre colonne
          int row_temp = A[idxm];
          row_shared[irs] = row_temp;
          t+= row_temp*B[icx];
        }
        else{
          //Se sto facendo la riga per le altre colonne allora prendo i valori dalla memoria shared
          t += row_shared[q + size_shared_vec * tc]*B[icx];
          //t += A[idxm]*B[icx];
          #ifdef _DEBUG
          printf("{Blocco %d} P%d-A[%d]: %f --- x[%d]: %f\n", row, tc, idxm, A[idxm], icx, B[icx]); 
          #endif
        }
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

__global__ void gpuMatrixVectorV1(int m, int k, int n, const float* A,
				const float* B, float* y) {
  __shared__ float aux[BD];
  int tc     = threadIdx.x;
  int row    = blockIdx.x;
  if (row < m) {
    
    // Starting address of indexing within matrix A
    for(int i = 0; i < n; i++){
      int idxm = row*k+tc;
      float t  = 0.0;
      int q = 0;
      aux[tc] = 0.0;
      for (int ic= tc;  ic<k; ic += blockDim.x) {
        int icx = i * n + ic;
        t += A[idxm]*B[icx];
        #ifdef _DEBUG
          printf("{Blocco %d} P%d-A[%d]: %f --- x[%d]: %f\n", row, tc, idxm, A[idxm], icx, B[icx]); 
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
  float* h_B = new float[k * n];
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
  std::cout << "\nMatrix B:" << std::endl;
  #endif
  for (int col = 0; col < k; ++col) {
    for(int row = 0; row < n; ++row){
      int idx = row * k + col;
      h_B[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
      #ifdef _DEBUG 
      std::cout << "|" << h_B[idx] << "";
      #endif
    }
    #ifdef _DEBUG 
    std::cout << "|"<< std::endl;
    #endif
  }

  memset(h_y, 0, m * n * sizeof(float));
  

  std::cout << "Matrix-vector product: 1D thread block version " << std::endl;
  std::cout << "Test case: [" << m  << "x" << k << "] x ["<< k << "x" << n << "]" << std::endl;
  std::cout << "m = " << m  << " | k = " << k << "| n = "<< n << std::endl;
// ---------------------- Device memory initialisation ---------------------- //

  float *d_A, *d_B, *d_y;

  checkCudaErrors(cudaMalloc((void**) &d_A, m * k * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_B, k * n * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_y, m * n * sizeof(float)));

  // Copy matrices from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, h_y, m * n * sizeof(float), cudaMemcpyHostToDevice));

  // ------------------------ Calculations on the CPU ------------------------- //
  
  float flopcnt=2.e-6*m*k*n;
  
  // Create the CUDA SDK timer.
  
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);
  
  
  timer->start();
  CpuMatrixVector(m, k, n, h_A, h_B, h_y);

  timer->stop();
  float cpuflops=flopcnt/ timer->getTime();
  std::cout << "  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;
  

// ------------------------ Calculations on the GPU ------------------------- //

  // Calculate the dimension of the grid of blocks (1D) needed to cover all
  // entries in the matrix and output vector
  const dim3 GRID_DIM(m,1);
  size_t smemSize = k * sizeof(float);
  float gpuflops;

  //printf("size of shared memory: %d\n", smemSize);
  
  timer->reset();
  timer->start();
  gpuMatrixVectorV1<<<GRID_DIM, BLOCK_DIM>>>(m, k, n, d_A, d_B, d_y);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  gpuflops=flopcnt/ timer->getTime();
  std::cout << "  GPU time global memory: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

  float* zero = new float[m * n];
  memset(zero, 0, m * n * sizeof(float));
  checkCudaErrors(cudaMemcpy(d_y, zero, m * n * sizeof(float), cudaMemcpyHostToDevice));
  
  printf("size of shared memory: %d\n", BD * BD2 * sizeof(float) + BD * sizeof(float));
  timer->reset();
  timer->start();
  cudaProfilerStart();
  gpuMatrixVectorV4<<<GRID_DIM, BLOCK_DIM>>>(m, k, n, d_A, d_B, d_y);
  cudaProfilerStop();
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  gpuflops=flopcnt/ timer->getTime();
  std::cout << "  GPU time shared memory: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

  checkCudaErrors(cudaMemcpy(d_y, zero, m * n * sizeof(float), cudaMemcpyHostToDevice));

  printf("size of shared memory: %d\n", BD * BD2 * sizeof(float) + BD * sizeof(float));
  timer->reset();
  timer->start();
  cudaProfilerStart();
  gpuMatrixMatrix<<<GRID_DIM, BLOCK_DIM>>>(m, k, n, d_A, d_B, d_y);
  cudaProfilerStop();
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  gpuflops=flopcnt/ timer->getTime();
  std::cout << "  GPU time shared memory: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

  

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

  #ifdef _DEBUG
  std::cout << "Matrix-vector product: CPU version " << std::endl;
  std::cout << "Test case: " << m  << " x " << n << std::endl;
  std::cout << "Matrix y_d: " << std::endl;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int idx = row * n + col;
      std::cout << "|" << h_y[idx] << "";
    }
    std::cout << "|" << std::endl;
  }
  #endif

  // Now let's check if the results are the same.
  float reldiff = 0.0f;
  float diff = 0.0f;
  
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float maxabs = std::max(std::abs(h_y[row + col * n]),std::abs(h_y_d[row + col * n]));
      if (maxabs == 0.0) maxabs=1.0;
      reldiff = std::max(reldiff, std::abs(h_y[row + col * n] - h_y_d[row + col * n])/maxabs);
      diff = std::max(diff, std::abs(h_y[row + col * n] - h_y_d[row + col * n]));
      //std::cout << row<<" "<<h_y[row]<<" "<<h_y_d[row] <<std::endl;
    }
  }
  std::cout << "Max diff = " << diff << "  Max rel diff = " << reldiff << std::endl;
  // Rel diff should be as close as possible to unit roundoff; float
  // corresponds to IEEE single precision, so unit roundoff is
  // 1.19e-07
  // 

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_y));
  
  delete[] h_A;
  delete[] h_B;
  delete[] h_y;
  delete[] h_y_d;
  
  return 0;
}
