#include <mpi.h>
#include "cuda_prova.h"


 
int main(int argc, char *argv[])
{
    
    /* It's important to put this call at the begining of the program, after variable declarations. */
    
    MPI_Init(&argc, &argv);
    int myRank, numProcs;

    

    /* Get the number of MPI processes and the rank of this process. */
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    printf("Hello from process %d of %d\n", myRank, numProcs);
    
    // ==== Call function 'call_me_maybe' from CUDA file multiply.cu: ==========
    printf("before CUDA\n");
    call_me_maybe();
    printf("after CUDA\n");
    /* ... */
    MPI_Finalize();
 
}