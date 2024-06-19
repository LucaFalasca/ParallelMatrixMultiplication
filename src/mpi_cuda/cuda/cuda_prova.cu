/* multiply.cu */
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_profiler_api.h>
//#include <helper_cuda.h> 
 
 __global__ void __multiply__ ()
 {
    
    /* ... */
 }
 
extern "C" void call_me_maybe()
{
   const dim3 GRID_DIM(1,1);
   const dim3 BLOCK_DIM(10);
/* ... Load CPU data into GPU buffers  */

   __multiply__<<<GRID_DIM, BLOCK_DIM>>>();
   //checkCudaErrors(cudaDeviceSynchronize());

/* ... Transfer data from GPU to CPU */
}