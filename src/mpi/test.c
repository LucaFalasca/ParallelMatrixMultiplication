#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int a[2];
    a[0] = 1;
    a[1] = 1;
    int c[2];
    c[0]=1;
    c[1]=1;
    int rank, size;
    MPI_Comm comm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    MPI_Reduce(a, c, 2, MPI_INT, MPI_SUM, 0, comm);

    printf("Process %d: c[0] = %d, c[1] = %d\n", rank, c[0], c[1]);

    MPI_Finalize();

    return 0;
}