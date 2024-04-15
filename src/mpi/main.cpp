#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "mat_mul.h"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int pg_row, pg_col, block_size;
    int row_a, col_a, row_b, col_b;
    char mat_a_path[128], mat_b_path[128], mat_c_path[128], mat_c_path_check[128];
    MPI_Comm comm;
    int rank, size;

    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*Get parameters from cmd*/
    if (argc < 12)
    {
        printf("Usage ./a.out <nrproc> <ncproc> <blocks> <matApath> <rowsA> <colsA> <matBpath> <rowsB> <colsB> <matCpath> <matCpath_check\n");
        exit(1);
    }

    // Process grid size
    pg_row = atoi(argv[1]);
    pg_col = atoi(argv[2]);


    // Block size
    block_size = atoi(argv[3]);

    // Matrix A data
    strcpy(mat_a_path, argv[4]);
    row_a = atoi(argv[5]);
    col_a = atoi(argv[6]);

    // Matrix B data
    strcpy(mat_b_path, argv[7]);
    row_b = atoi(argv[8]);
    col_b = atoi(argv[9]);

    // Matric C data
    strcpy(mat_c_path, argv[10]);
    strcpy(mat_c_path_check, argv[11]);

    /*Check size compatibility for matrix multiply*/
    if (col_a != row_b)
    {
        printf("Incompatible matrix size for multiplication c1!=r2\n");
        exit(1);
    }

    double start = MPI_Wtime();

    parallel_matrix_multiplication(pg_row, pg_col, block_size, mat_a_path, row_a, col_a, mat_b_path, row_b, col_b, mat_c_path, mat_c_path_check);

    if(rank==0){
        double end = MPI_Wtime();
        printf("Measured performance:\n");
        printf("\tGFLOPS: %lf\n", (2.0 * row_a * col_a * col_b) / (end - start) / 1e9);
        printf("\tElapsed %lf ms\n", (end - start) * 1000);
    }
    

    MPI_Finalize();
    return 0;
}