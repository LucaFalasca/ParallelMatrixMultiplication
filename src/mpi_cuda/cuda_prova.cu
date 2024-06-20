/* multiply.cu */
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h> 
 
 __global__ void __multiply__ (float *t)
 {
    t[0] = 150;
   printf("prova\n");
    /* ... */
 }
 
extern "C" void call_me_maybe()
{
   const dim3 GRID_DIM(1,1);
   const dim3 BLOCK_DIM(10);
/* ... Load CPU data into GPU buffers  */
   float *t;
   cudaMalloc((void**) &t, 1 * sizeof(float));

   __multiply__<<<GRID_DIM, BLOCK_DIM>>>(t);

   float *res;
   //faccio la malloc
   //res = (float *)malloc(1*sizeof(float));
   res = new float[1];
   res[0] = 0;
   cudaMemcpy(res, t, 1*sizeof(float),cudaMemcpyDeviceToHost);
   checkCudaErrors(cudaDeviceSynchronize());
   printf("res: %f\n", res[0]);

   cudaFree(t);
   //delete[] res;
   

/* ... Transfer data from GPU to CPU */
}