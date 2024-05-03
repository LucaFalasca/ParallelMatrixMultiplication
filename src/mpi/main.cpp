#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "mat_mul.h"
#include <iostream>

void write_result(int num_procs, int pr, int pc, int block_size, int row_a, int col_a, int row_b, int col_b, double gflops, double elapsed_time, float max_diff, float max_rel_diff);

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
        double gflops = ((2.0 * row_a * col_a * col_b) / (end - start)) / 1e9;
        double elapsed_time = (end - start) * 1000;
        
        //printf("Checking result...\n");
        float err[2] = {0.0, 0.0};
        //float *err=check_result(mat_a_path, mat_b_path, mat_c_path, mat_c_path_check, row_a, col_a, col_b);
        
        std::cout << "Measured performance:" << std::endl;
        std::cout << "\tGFLOPS: " << gflops << std::endl;
        std::cout << "\tElapsed " << elapsed_time <<"ms" << std::endl;
        std::cout << "\tMax diff: " << err[0] << std::endl;
        std::cout << "\tMax relative diff: "<< err[1] << std::endl;

        std::cout << "Writing data on csv..." << std::endl;
        write_result(size, pg_row, pg_col, block_size, row_a, col_a, row_b, col_b, gflops, elapsed_time, err[0], err[1]);
        
        std::cout << "Resetting matrix C..." << std::endl;
        reset_matrix_c(mat_c_path, mat_c_path_check);
    }
    

    MPI_Finalize();
    return 0;
}

void write_result(int num_procs, int pr, int pc, int block_size, int row_a, int col_a, int row_b, int col_b, double gflops, double elapsed_time, float max_diff, float max_rel_diff) {
    FILE *fp;
    bool file_exists = false;
    fp = fopen("data/out/results.csv", "a+"); // "a+" mode opens for reading and appending

    if (fp == NULL) {
        printf("Error opening file!\n");
        return;
    }

    // Check if file is empty
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        file_exists = false;
    } else {
        file_exists = true;
    }

    // Write header if file doesn't exist
    if (!file_exists) {
        fprintf(fp, "num_proc, num_proc_row, num_proc_col, block_size, mat_A_rows, mat_A_cols, mat_B_rows, mat_B_cols, gflops, elapsed_time, max_diff, max_rel_diff\n");
    }

    fprintf(fp, "%d, %d, %d, %d, %d, %d, %d, %d, %lf, %lf, %f, %e\n", 
            num_procs, pr, pc, block_size, row_a, col_a, row_b, col_b, gflops, elapsed_time, max_diff, max_rel_diff);

    fclose(fp);
}
