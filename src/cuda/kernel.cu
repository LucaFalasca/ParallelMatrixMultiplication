#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_timer.h>
#include <helper_cuda.h>



#define BD 512
#define BD2 12
#define COLS 11
//#define _DEBUG

const dim3 BLOCK_DIM(BD);

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


__global__ void gpuMatrixMatrixV4(int m, int k, int n, const float* A,
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
          int icx = (i + j) + n * ic;
          //printf("icx %d --- icx2 %d\n", icx, icx2);
          
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
      /*
      for(int j = 0; j < n_cols; j++){
        for (int s2=BD/2; s2 >=32; s2 >>=1){
          if (tc<s2)
            aux[j][tc] += aux[j][tc+s2]; 
          __syncthreads();
        }
      }*/

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

extern "C" void kernel(int m, int k, int n, float* A, float* B, float* y){  
  // ----------------------- Host memory initialisation ----------------------- //
  

  float* h_A = A;
  float* h_B = B;
  float* h_y = y;
  float* h_y_d = new float[m * n];
  #ifdef _PRINT_MATRIX
  std::cout << "Test case: " << m  << " x " << n << std::endl;
  std::cout << "Matrix A: " << std::endl;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int idx = row * n + col;
      std::cout << "|" << h_A[idx] << "";
    }
    std::cout << "|" << std::endl;
  }

  std::cout << "Test case: " << m  << " x " << n << std::endl;
  std::cout << "Matrix B: " << std::endl;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int idx = row * n + col;
      std::cout << "|" << h_B[idx] << "";
    }
    std::cout << "|" << std::endl;
  }


  std::cout << "Test case: " << m  << " x " << n << std::endl;
  std::cout << "Matrix C: " << std::endl;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int idx = row * n + col;
      std::cout << "|" << h_y[idx] << "";
    }
    std::cout << "|" << std::endl;
  }
  #endif

  srand(123456);

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

  float* zero = new float[m * n];
  memset(zero, 0, m * n * sizeof(float));
  checkCudaErrors(cudaMemcpy(d_y, zero, m * n * sizeof(float), cudaMemcpyHostToDevice));

  // ------------------------ Calculations on the CPU ------------------------- //
  
  float flopcnt=2.e-6*m*k*n;
  
  // Create the CUDA SDK timer.
  
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

// ------------------------ Calculations on the GPU ------------------------- //

  // Calculate the dimension of the grid of blocks (1D) needed to cover all
  // entries in the matrix and output vector
  const dim3 GRID_DIM(m,1);
  float gpuflops;
  printf("size of shared memory: %d\n", BD * COLS * sizeof(float) + BD * sizeof(float));
  timer->reset();
  timer->start();
  gpuMatrixMatrixV4<<<GRID_DIM, BLOCK_DIM>>>(m, k, n, d_A, d_B, d_y);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  gpuflops=flopcnt/ timer->getTime();
  std::cout << "  GPU time V4: " << timer->getTime() << " ms." << " GFLOPS " << gpuflops<<std::endl;

  

  // Download the resulting vector d_y from the device and store it in h_y_d.
  checkCudaErrors(cudaMemcpy(h_y, d_y, m*n*sizeof(float),cudaMemcpyDeviceToHost));

  //print h_y_d
  #ifdef _PRINT_MATRIX
  std::cout << "Test case: " << m  << " x " << n << std::endl;
  std::cout << "Matrix result: " << std::endl;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int idx = row * n + col;
      std::cout << "|" << h_y[idx] << "";
    }
    std::cout << "|" << std::endl;
  }
  #endif

}