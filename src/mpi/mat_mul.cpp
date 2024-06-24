#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "mat_mul.h"
#include <cstdlib>
#include "../cuda/kernel.h"

void parallel_matrix_multiplication(int pg_row, int pg_col, int block_size, char *mat_a_path, int row_a, int col_a, char *mat_b_path, int row_b, int col_b, char *mat_c_path, char *mat_c_path_check, int version)
{
    if(version==0){
        printf("Running parallel naive matrix multiplication\n");
        parallel_matrix_multiplication_naive(pg_row, pg_col, block_size, mat_a_path, row_a, col_a, mat_b_path, row_b, col_b, mat_c_path, mat_c_path_check);
    }
    else if(version==1){
        printf("Running parallel blocked matrix multiplication\n");
        parallel_matrix_multiplication_blocked(pg_row, pg_col, block_size, mat_a_path, row_a, col_a, mat_b_path, row_b, col_b, mat_c_path, mat_c_path_check);
    }
    else if(version==2){
        printf("Running parallel accelerated matrix multiplication\n");
        parallel_matrix_multiplication_accelerated(pg_row, pg_col, block_size, mat_a_path, row_a, col_a, mat_b_path, row_b, col_b, mat_c_path, mat_c_path_check);
    }
    else{
        printf("Invalid version\n");
        exit(1);
    }
}

void parallel_matrix_multiplication_accelerated(int pg_row, int pg_col, int block_size, char *mat_a_path, int row_a, int col_a, char *mat_b_path, int row_b, int col_b, char *mat_c_path, char *mat_c_path_check)
{
    float *partial_res;
    struct submat_info *submat_A_info, *submat_B_info, *submat_C_info;
    struct comm_info *comm_info, *row_comm_info, *col_comm_info, *row_leader_comm_info;

    submat_A_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_A_info == NULL)
    {
        printf("Error in memory allocation for submat_A_info\n");
        exit(1);
    }
    submat_B_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_B_info == NULL)
    {
        printf("Error in memory allocation for submat_B_info\n");
        exit(1);
    }

    submat_C_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_C_info == NULL)
    {
        printf("Error in memory allocation for matrix C submat\n");
        exit(1);
    }
    memset(submat_C_info, 0, sizeof(struct submat_info));

    comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (comm_info == NULL)
    {
        printf("Error in memory allocation for comm_info\n");
        exit(1);
    }
    row_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (row_comm_info == NULL)
    {
        printf("Error in memory allocation for row_comm_info\n");
        exit(1);
    }
    col_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (col_comm_info == NULL)
    {
        printf("Error in memory allocation for col_comm_info\n");
        exit(1);
    }
    row_leader_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (row_leader_comm_info == NULL)
    {
        printf("Error in memory allocation for row_leader_comm_info\n");
        exit(1);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &(comm_info->comm));
    MPI_Comm_rank(MPI_COMM_WORLD, &(comm_info->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(comm_info->size));

    /*Check size compatibility for process grid*/
    if ((pg_row * pg_col) != comm_info->size)
    {
        printf("Process grid size incompatible with number of processes spawned\n");
        exit(1);
    }

#ifdef AUDIT
    if (comm_info->rank == 0)
    {
        printf("AUDIT -> Number of processes: %d\n", comm_info->size);
        printf("AUDIT -> Process grid size: %d x %d\n", pg_row, pg_col);
        printf("AUDIT -> Block size: %d x %d\n", block_size, block_size);
        printf("AUDIT -> Matrix A path %s size: %d x %d\n", mat_a_path, row_a, col_a);
        printf("AUDIT -> Matrix B path %s size: %d x %d\n", mat_b_path, row_b, col_b);
        printf("AUDIT -> Matrix C path %s size: %d x %d\n", mat_c_path, row_a, col_b);
        printf("AUDIT -> Matrix C path check %s size: %d x %d\n", mat_c_path_check, row_a, col_b);
    }
#endif

    // Each process calculates its position in the process grid
    set_proc_grid_info(pg_col, comm_info);

    /*Create row communicator, each row of processes get the same color as it contributes to the same row of the result matrix
    in this manner we build a leader-follower architecture to compute the effective row of the result matrix, each follower
    computes only only A*B and send the result to the leader that will sum the partial results and perform C+=A*B.
    All the followers work in different zones of the result matrix
    */
    create_row_comm(pg_col, comm_info, row_comm_info);
    create_col_comm(pg_row, comm_info, col_comm_info);

    // Create a communicator with only the row leaders which have to perform the MPI I/O ops on the result matrix file
    create_row_leader_comm(pg_row, pg_col, comm_info, row_leader_comm_info);

    // Distribute the matrix A
    compute_block_info(row_a, col_a, block_size, block_size, pg_row, pg_col, comm_info, submat_A_info);
    block_cyclic_distribution(mat_a_path, row_a, col_a, block_size, pg_row, pg_col, submat_A_info, comm_info);

#ifdef DEBUG_ELEMENT
    MPI_Barrier(comm_info->comm);
    for (int i = 0; i < (submat_A_info->submat_row) * (submat_A_info->submat_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submat of A: %f\n", comm_info->pg_row_idx, comm_info->pg_col_idx, submat_A_info->submat[i]);
    }
#endif

    // Distribute the matrix B
    compute_row_block_info(row_b, col_b, block_size, pg_row, pg_col, comm_info, submat_B_info);
    row_block_cyclic_distribution(mat_b_path, row_b, col_b, block_size, pg_row, pg_col, submat_B_info, comm_info);

#ifdef DEBUG_ELEMENT
    MPI_Barrier(comm_info->comm);
    for (int i = 0; i < (submat_B_info->submat_row) * (submat_B_info->submat_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submat of B: %f\n", comm_info->pg_row_idx, comm_info->pg_col_idx, submat_B_info->submat[i]);
    }
#endif

    int submat_A_row = submat_A_info->submat_row;
    int submat_A_col = submat_A_info->submat_col;
    int submat_B_col = submat_B_info->submat_col;

    // Allocate partial result submatrix
    //partial_res = (float *)malloc(submat_A_row * submat_B_col * sizeof(float));
    partial_res = (float *)calloc(submat_A_row * submat_B_col, sizeof(float));
    if (partial_res == NULL)
    {
        printf("Error in memory allocation for partial matrix result\n");
        exit(1);
    }

    // Only the process leader of the row will read C
    if (row_leader_comm_info->comm != MPI_COMM_NULL)
    {
        compute_row_block_info(row_a, col_b, block_size, 1, pg_row, row_leader_comm_info, submat_C_info);
        row_block_cyclic_distribution(mat_c_path, row_a, col_b, block_size, 1, pg_row, submat_C_info, row_leader_comm_info);
        memcpy(partial_res, submat_C_info->submat, submat_C_info->submat_row * submat_C_info->submat_col * sizeof(float));

#ifdef DEBUG_ELEMENT
        MPI_Barrier(row_leader_comm_info->comm);
        for (int i = 0; i < submat_C_info->submat_row * submat_C_info->submat_col; i++)
        {
            printf("Rank %d in grid (%d, %d) has element %f of submat of C\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, partial_res[i]);
        }
#endif
    }

#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    if (row_leader_comm_info->comm == MPI_COMM_NULL)
        printf("Rank %d in grid (%d, %d) belongs to row comm %d, has %dx%d submat of A, %dx%d submat of B and %dx%d partial result\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, comm_info->pg_row_idx, submat_A_row, submat_A_col, submat_B_info->submat_row, submat_B_col, submat_A_row, submat_B_col);
    else
        printf("Rank %d in grid (%d, %d) is leader of row comm %d, has %dx%d submat of A, %dx%d submat of B, %dx%d submat of C and %dx%d partial result\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, comm_info->pg_row_idx, submat_A_row, submat_A_col, submat_B_info->submat_row, submat_B_col, submat_C_info->submat_row, submat_C_info->submat_col, submat_A_row, submat_B_col);
#endif

    // Perform multiplication of submat
    int m = submat_A_info->submat_row;
    int k = submat_A_info->submat_col;
    int n = submat_B_info->submat_col;
    printf("N: %d", n);
    float *A = submat_A_info->submat;
    float *B = submat_B_info->submat;
    kernel(m, k, n, submat_A_info->submat, submat_B_info->submat, partial_res);
    
    // Free submat a and b
    free(submat_A_info);
    free(submat_B_info);

#ifdef DEBUG_ELEMEN
    MPI_Barrier(comm_info->comm);
    for(int i=0; i<submat_C_info->submat_row*submat_C_info->submat_col; i++)
        printf("Rank %d element in pos %d before reduce = %f\n", comm_info->rank, i, partial_res[i]);
#endif 

    //Reduce reduce on row leaders
    MPI_Reduce(partial_res, submat_C_info->submat, submat_A_row * submat_B_col, MPI_FLOAT, MPI_SUM, 0, row_comm_info->comm);


#ifdef DEBUG_ELEMEN
    MPI_Barrier(comm_info->comm);
    if(row_leader_comm_info->comm != MPI_COMM_NULL){
        for(int i=0; i<submat_C_info->submat_row*submat_C_info->submat_col; i++)
            printf("Rank %d element in pos %d after reduce = %f\n", comm_info->rank, i, submat_C_info->submat[i]);
    }
#endif

    // Free partial result matrix
    free(partial_res);

    // Leader write result
    if (row_leader_comm_info->comm != MPI_COMM_NULL)
    {
        block_cyclic_write_result(mat_c_path, row_a, col_b, block_size, 1, pg_row, submat_C_info, row_leader_comm_info);
    }

    // Free submat C
    free(submat_C_info);

#ifdef CHECK_RESULT
    MPI_Barrier(comm_info->comm);
    if (comm_info->rank == 0)
    {
        printf("Rank 0 checking result...\n");
        check_result(mat_a_path, mat_b_path, mat_c_path, mat_c_path_check, row_a, col_a, col_b);
    }
#endif
}

void parallel_matrix_multiplication_naive(int pg_row, int pg_col, int block_size, char *mat_a_path, int row_a, int col_a, char *mat_b_path, int row_b, int col_b, char *mat_c_path, char *mat_c_path_check)
{
    float *partial_res;
    struct submat_info *submat_A_info, *submat_B_info, *submat_C_info;
    struct comm_info *comm_info, *row_comm_info, *col_comm_info, *row_leader_comm_info;

    submat_A_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_A_info == NULL)
    {
        printf("Error in memory allocation for submat_A_info\n");
        exit(1);
    }
    submat_B_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_B_info == NULL)
    {
        printf("Error in memory allocation for submat_B_info\n");
        exit(1);
    }

    submat_C_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_C_info == NULL)
    {
        printf("Error in memory allocation for matrix C submat\n");
        exit(1);
    }
    memset(submat_C_info, 0, sizeof(struct submat_info));

    comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (comm_info == NULL)
    {
        printf("Error in memory allocation for comm_info\n");
        exit(1);
    }
    row_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (row_comm_info == NULL)
    {
        printf("Error in memory allocation for row_comm_info\n");
        exit(1);
    }
    col_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (col_comm_info == NULL)
    {
        printf("Error in memory allocation for col_comm_info\n");
        exit(1);
    }
    row_leader_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (row_leader_comm_info == NULL)
    {
        printf("Error in memory allocation for row_leader_comm_info\n");
        exit(1);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &(comm_info->comm));
    MPI_Comm_rank(MPI_COMM_WORLD, &(comm_info->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(comm_info->size));

    /*Check size compatibility for process grid*/
    if ((pg_row * pg_col) != comm_info->size)
    {
        printf("Process grid size incompatible with number of processes spawned\n");
        exit(1);
    }

#ifdef AUDIT
    if (comm_info->rank == 0)
    {
        printf("AUDIT -> Number of processes: %d\n", comm_info->size);
        printf("AUDIT -> Process grid size: %d x %d\n", pg_row, pg_col);
        printf("AUDIT -> Block size: %d x %d\n", block_size, block_size);
        printf("AUDIT -> Matrix A path %s size: %d x %d\n", mat_a_path, row_a, col_a);
        printf("AUDIT -> Matrix B path %s size: %d x %d\n", mat_b_path, row_b, col_b);
        printf("AUDIT -> Matrix C path %s size: %d x %d\n", mat_c_path, row_a, col_b);
        printf("AUDIT -> Matrix C path check %s size: %d x %d\n", mat_c_path_check, row_a, col_b);
    }
#endif

    // Each process calculates its position in the process grid
    set_proc_grid_info(pg_col, comm_info);

    /*Create row communicator, each row of processes get the same color as it contributes to the same row of the result matrix
    in this manner we build a leader-follower architecture to compute the effective row of the result matrix, each follower
    computes only only A*B and send the result to the leader that will sum the partial results and perform C+=A*B.
    All the followers work in different zones of the result matrix
    */
    create_row_comm(pg_col, comm_info, row_comm_info);
    create_col_comm(pg_row, comm_info, col_comm_info);

    // Create a communicator with only the row leaders which have to perform the MPI I/O ops on the result matrix file
    create_row_leader_comm(pg_row, pg_col, comm_info, row_leader_comm_info);

    // Distribute the matrix A
    compute_block_info(row_a, col_a, block_size, block_size, pg_row, pg_col, comm_info, submat_A_info);
    block_cyclic_distribution(mat_a_path, row_a, col_a, block_size, pg_row, pg_col, submat_A_info, comm_info);

#ifdef DEBUG_ELEMENT
    MPI_Barrier(comm_info->comm);
    for (int i = 0; i < (submat_A_info->submat_row) * (submat_A_info->submat_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submat of A: %f\n", comm_info->pg_row_idx, comm_info->pg_col_idx, submat_A_info->submat[i]);
    }
#endif

    // Distribute the matrix B
    compute_row_block_info(row_b, col_b, block_size, pg_row, pg_col, comm_info, submat_B_info);
    row_block_cyclic_distribution(mat_b_path, row_b, col_b, block_size, pg_row, pg_col, submat_B_info, comm_info);

#ifdef DEBUG_ELEMENT
    MPI_Barrier(comm_info->comm);
    for (int i = 0; i < (submat_B_info->submat_row) * (submat_B_info->submat_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submat of B: %f\n", comm_info->pg_row_idx, comm_info->pg_col_idx, submat_B_info->submat[i]);
    }
#endif

    int submat_A_row = submat_A_info->submat_row;
    int submat_A_col = submat_A_info->submat_col;
    int submat_B_col = submat_B_info->submat_col;

    // Allocate partial result submatrix
    //partial_res = (float *)malloc(submat_A_row * submat_B_col * sizeof(float));
    partial_res = (float *)malloc(submat_A_row * submat_B_col*sizeof(float));
    if (partial_res == NULL)
    {
        printf("Error in memory allocation for partial matrix result\n");
        exit(1);
    }

    // Only the process leader of the row will read C
    if (row_leader_comm_info->comm != MPI_COMM_NULL)
    {
        compute_row_block_info(row_a, col_b, block_size, 1, pg_row, row_leader_comm_info, submat_C_info);
        row_block_cyclic_distribution(mat_c_path, row_a, col_b, block_size, 1, pg_row, submat_C_info, row_leader_comm_info);
        memcpy(partial_res, submat_C_info->submat, submat_C_info->submat_row * submat_C_info->submat_col * sizeof(float));

#ifdef DEBUG_ELEMENT
        MPI_Barrier(row_leader_comm_info->comm);
        for (int i = 0; i < submat_C_info->submat_row * submat_C_info->submat_col; i++)
        {
            printf("Rank %d in grid (%d, %d) has element %f of submat of C\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, partial_res[i]);
        }
#endif
    }

#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    if (row_leader_comm_info->comm == MPI_COMM_NULL)
        printf("Rank %d in grid (%d, %d) belongs to row comm %d, has %dx%d submat of A, %dx%d submat of B and %dx%d partial result\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, comm_info->pg_row_idx, submat_A_row, submat_A_col, submat_B_info->submat_row, submat_B_col, submat_A_row, submat_B_col);
    else
        printf("Rank %d in grid (%d, %d) is leader of row comm %d, has %dx%d submat of A, %dx%d submat of B, %dx%d submat of C and %dx%d partial result\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, comm_info->pg_row_idx, submat_A_row, submat_A_col, submat_B_info->submat_row, submat_B_col, submat_C_info->submat_row, submat_C_info->submat_col, submat_A_row, submat_B_col);
#endif

    // Perform multiplication of submat
    if(row_leader_comm_info->comm == MPI_COMM_NULL){
        matrix_multiply(submat_A_info->submat, submat_B_info->submat, partial_res, submat_A_row, submat_A_col, submat_B_col, true);
    }
    else{
        matrix_multiply(submat_A_info->submat, submat_B_info->submat, partial_res, submat_A_row, submat_A_col, submat_B_col, false);
    }

    // Free submat a and b
    free(submat_A_info);
    free(submat_B_info);

#ifdef DEBUG_ELEMEN
    MPI_Barrier(comm_info->comm);
    for(int i=0; i<submat_C_info->submat_row*submat_C_info->submat_col; i++)
        printf("Rank %d element in pos %d before reduce = %f\n", comm_info->rank, i, partial_res[i]);
#endif 

    //Reduce reduce on row leaders
    MPI_Reduce(partial_res, submat_C_info->submat, submat_A_row * submat_B_col, MPI_FLOAT, MPI_SUM, 0, row_comm_info->comm);


#ifdef DEBUG_ELEMEN
    MPI_Barrier(comm_info->comm);
    if(row_leader_comm_info->comm != MPI_COMM_NULL){
        for(int i=0; i<submat_C_info->submat_row*submat_C_info->submat_col; i++)
            printf("Rank %d element in pos %d after reduce = %f\n", comm_info->rank, i, submat_C_info->submat[i]);
    }
#endif

    // Free partial result matrix
    free(partial_res);

    // Leader write result
    if (row_leader_comm_info->comm != MPI_COMM_NULL)
    {
        block_cyclic_write_result(mat_c_path, row_a, col_b, block_size, 1, pg_row, submat_C_info, row_leader_comm_info);
    }

    // Free submat C
    free(submat_C_info);

#ifdef CHECK_RESULT
    MPI_Barrier(comm_info->comm);
    if (comm_info->rank == 0)
    {
        printf("Rank 0 checking result...\n");
        check_result(mat_a_path, mat_b_path, mat_c_path, mat_c_path_check, row_a, col_a, col_b);
    }
#endif
}


void parallel_matrix_multiplication_blocked(int pg_row, int pg_col, int block_size, char *mat_a_path, int row_a, int col_a, char *mat_b_path, int row_b, int col_b, char *mat_c_path, char *mat_c_path_check)
{
    float *partial_res;
    struct submat_info *submat_A_info, *submat_B_info, *submat_C_info;
    struct comm_info *comm_info, *row_comm_info, *col_comm_info, *row_leader_comm_info;

    submat_A_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_A_info == NULL)
    {
        printf("Error in memory allocation for submat_A_info\n");
        exit(1);
    }
    submat_B_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_B_info == NULL)
    {
        printf("Error in memory allocation for submat_B_info\n");
        exit(1);
    }

    submat_C_info = (struct submat_info *)malloc(sizeof(struct submat_info));
    if (submat_C_info == NULL)
    {
        printf("Error in memory allocation for matrix C submat\n");
        exit(1);
    }
    memset(submat_C_info, 0, sizeof(struct submat_info));

    comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (comm_info == NULL)
    {
        printf("Error in memory allocation for comm_info\n");
        exit(1);
    }
    row_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (row_comm_info == NULL)
    {
        printf("Error in memory allocation for row_comm_info\n");
        exit(1);
    }
    col_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (col_comm_info == NULL)
    {
        printf("Error in memory allocation for col_comm_info\n");
        exit(1);
    }
    row_leader_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (row_leader_comm_info == NULL)
    {
        printf("Error in memory allocation for row_leader_comm_info\n");
        exit(1);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &(comm_info->comm));
    MPI_Comm_rank(MPI_COMM_WORLD, &(comm_info->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(comm_info->size));

    /*Check size compatibility for process grid*/
    if ((pg_row * pg_col) != comm_info->size)
    {
        printf("Process grid size incompatible with number of processes spawned\n");
        exit(1);
    }

#ifdef AUDIT
    if (comm_info->rank == 0)
    {
        printf("AUDIT -> Number of processes: %d\n", comm_info->size);
        printf("AUDIT -> Process grid size: %d x %d\n", pg_row, pg_col);
        printf("AUDIT -> Block size: %d x %d\n", block_size, block_size);
        printf("AUDIT -> Matrix A path %s size: %d x %d\n", mat_a_path, row_a, col_a);
        printf("AUDIT -> Matrix B path %s size: %d x %d\n", mat_b_path, row_b, col_b);
        printf("AUDIT -> Matrix C path %s size: %d x %d\n", mat_c_path, row_a, col_b);
        printf("AUDIT -> Matrix C path check %s size: %d x %d\n", mat_c_path_check, row_a, col_b);
    }
#endif

    // Each process calculates its position in the process grid
    set_proc_grid_info(pg_col, comm_info);

    /*Create row communicator, each row of processes get the same color as it contributes to the same row of the result matrix
    in this manner we build a leader-follower architecture to compute the effective row of the result matrix, each follower
    computes only only A*B and send the result to the leader that will sum the partial results and perform C+=A*B.
    All the followers work in different zones of the result matrix
    */
    create_row_comm(pg_col, comm_info, row_comm_info);
    create_col_comm(pg_row, comm_info, col_comm_info);

    // Create a communicator with only the row leaders which have to perform the MPI I/O ops on the result matrix file
    create_row_leader_comm(pg_row, pg_col, comm_info, row_leader_comm_info);

    // Distribute the matrix A
    compute_block_info(row_a, col_a, block_size, block_size, pg_row, pg_col, comm_info, submat_A_info);
    printf("Rank %d will receive matrix of size %dx%d", submat_A_info->submat_row, submat_A_info->submat_col);
    exit(0);
    block_cyclic_distribution(mat_a_path, row_a, col_a, block_size, pg_row, pg_col, submat_A_info, comm_info);

#ifdef DEBUG_ELEMENT
    MPI_Barrier(comm_info->comm);
    for (int i = 0; i < (submat_A_info->submat_row) * (submat_A_info->submat_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submat of A: %f\n", comm_info->pg_row_idx, comm_info->pg_col_idx, submat_A_info->submat[i]);
    }
#endif

    // Distribute the matrix B
    compute_row_block_info(row_b, col_b, block_size, pg_row, pg_col, comm_info, submat_B_info);
    row_block_cyclic_distribution(mat_b_path, row_b, col_b, block_size, pg_row, pg_col, submat_B_info, comm_info);

#ifdef DEBUG_ELEMENT
    MPI_Barrier(comm_info->comm);
    for (int i = 0; i < (submat_B_info->submat_row) * (submat_B_info->submat_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submat of B: %f\n", comm_info->pg_row_idx, comm_info->pg_col_idx, submat_B_info->submat[i]);
    }
#endif

    int submat_A_row = submat_A_info->submat_row;
    int submat_A_col = submat_A_info->submat_col;
    int submat_B_col = submat_B_info->submat_col;

    // Allocate partial result submatrix
    //partial_res = (float *)malloc(submat_A_row * submat_B_col * sizeof(float));
    partial_res = (float *)calloc(submat_A_row * submat_B_col, sizeof(float));
    if (partial_res == NULL)
    {
        printf("Error in memory allocation for partial matrix result\n");
        exit(1);
    }

    // Only the process leader of the row will read C
    if (row_leader_comm_info->comm != MPI_COMM_NULL)
    {
        compute_row_block_info(row_a, col_b, block_size, 1, pg_row, row_leader_comm_info, submat_C_info);
        row_block_cyclic_distribution(mat_c_path, row_a, col_b, block_size, 1, pg_row, submat_C_info, row_leader_comm_info);
        memcpy(partial_res, submat_C_info->submat, submat_C_info->submat_row * submat_C_info->submat_col * sizeof(float));

#ifdef DEBUG_ELEMENT
        MPI_Barrier(row_leader_comm_info->comm);
        for (int i = 0; i < submat_C_info->submat_row * submat_C_info->submat_col; i++)
        {
            printf("Rank %d in grid (%d, %d) has element %f of submat of C\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, partial_res[i]);
        }
#endif
    }

#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    if (row_leader_comm_info->comm == MPI_COMM_NULL)
        printf("Rank %d in grid (%d, %d) belongs to row comm %d, has %dx%d submat of A, %dx%d submat of B and %dx%d partial result\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, comm_info->pg_row_idx, submat_A_row, submat_A_col, submat_B_info->submat_row, submat_B_col, submat_A_row, submat_B_col);
    else
        printf("Rank %d in grid (%d, %d) is leader of row comm %d, has %dx%d submat of A, %dx%d submat of B, %dx%d submat of C and %dx%d partial result\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, comm_info->pg_row_idx, submat_A_row, submat_A_col, submat_B_info->submat_row, submat_B_col, submat_C_info->submat_row, submat_C_info->submat_col, submat_A_row, submat_B_col);
#endif

    // Perform multiplication of submat
    column_blocked_matrix_multiply(submat_A_info->submat, submat_B_info->submat, partial_res, submat_A_row, submat_A_col, submat_B_col);

    // Free submat a and b
    free(submat_A_info);
    free(submat_B_info);

#ifdef DEBUG_ELEMEN
    MPI_Barrier(comm_info->comm);
    for(int i=0; i<submat_C_info->submat_row*submat_C_info->submat_col; i++)
        printf("Rank %d element in pos %d before reduce = %f\n", comm_info->rank, i, partial_res[i]);
#endif 

    //Reduce reduce on row leaders
    MPI_Reduce(partial_res, submat_C_info->submat, submat_A_row * submat_B_col, MPI_FLOAT, MPI_SUM, 0, row_comm_info->comm);


#ifdef DEBUG_ELEMEN
    MPI_Barrier(comm_info->comm);
    if(row_leader_comm_info->comm != MPI_COMM_NULL){
        for(int i=0; i<submat_C_info->submat_row*submat_C_info->submat_col; i++)
            printf("Rank %d element in pos %d after reduce = %f\n", comm_info->rank, i, submat_C_info->submat[i]);
    }
#endif

    // Free partial result matrix
    free(partial_res);

    // Leader write result
    if (row_leader_comm_info->comm != MPI_COMM_NULL)
    {
        block_cyclic_write_result(mat_c_path, row_a, col_b, block_size, 1, pg_row, submat_C_info, row_leader_comm_info);
    }

    // Free submat C
    free(submat_C_info);

#ifdef CHECK_RESULT
    MPI_Barrier(comm_info->comm);
    if (comm_info->rank == 0)
    {
        printf("Rank 0 checking result...\n");
        check_result(mat_a_path, mat_b_path, mat_c_path, mat_c_path_check, row_a, col_a, col_b);
    }
#endif
}


void column_blocked_matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2){
    // For each chunk of columns
    for (int col_chunk = 0; col_chunk < c2; col_chunk += 16){
        // For each row in that chunk of columns...
        for (int row = 0; row < r1; row++){
            // For each block of elements in this row of this column chunk
            // Solve for 16 elements at a time
            for (int tile = 0; tile < c1; tile += 16){
                // For each row in the tile
                for (int tile_row = 0; tile_row < 16; tile_row++){
                    // Solve for each element in this tile row
                    for (int idx = 0; idx < 16; idx++){
                        if ((tile + tile_row < c1) && (col_chunk + idx < c2)) {
                                        res[row * c2 + col_chunk + idx] +=
                                            mat1[row * c1 + tile + tile_row] *
                                            mat2[(tile + tile_row) * c2 + col_chunk + idx];
                        }
                    }
                }
            }
        }
    }
}

void matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2, bool res_zero)
{
    int i, j, k;
    for (i = 0; i < r1; ++i)//N
    {
        for (j = 0; j < c2; ++j)//M
        {
            if(res_zero)
               res[i * c2 + j] = 0;
            
            for (k = 0; k < c1; ++k)//K
            {
                res[i * c2 + j] += mat1[i * c1 + k] * mat2[k * c2 + j];
            }
        }
    }
}

// Distribuzione della matrice in blocchi con metodo block cyclic
void block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info)
{
    MPI_Status status;
    MPI_Datatype mat_darray;
    MPI_File mat_file;
    int dims[2] = {row, col};                                         // Dimensione matrice originale
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC}; // Metodo di distribuzione dei blocchi
    int dargs[2] = {block_size, block_size};                          // Dimensione dei blocchi
    int proc_dims[2] = {pg_row, pg_col};                              // Dimensione della griglia di processi
    float *recv_block;

    recv_block = (float *)malloc(submat_info->submat_row * submat_info->submat_col * sizeof(float));
    if (recv_block == NULL)
    {
        printf("Error in memory allocation for recv_block in block_cyclic_distribution\n");
        exit(1);
    }
    // recv_block = memset(recv_block, 0, submat_info->submat_row * submat_info->submat_col * sizeof(float));

    /*
    Creazione del tipo di dato per la matrice distribuita, ogni processo vedrà solo la sua porzione di matrice,
    la porzione viene definita tramite block cyclic distribution
    */
    MPI_Type_create_darray(comm_info->size, comm_info->rank, 2, dims, distribs, dargs, proc_dims, MPI_ORDER_C, MPI_FLOAT, &mat_darray);
    MPI_Type_commit(&mat_darray);

// Apertura collettiva del file
#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    if (comm_info->rank == 0)
    {
        printf("DEBUG -> Opening file %s\n", mat_path);
    }
#endif
    MPI_File_open(comm_info->comm, mat_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_file);
    if (mat_file == MPI_FILE_NULL)
    {
        printf("Error opening file %s in block cyclic distribution\n", mat_path);
        exit(1);
    }

    // Ogni processo ha una visione della matrice specificata dal darray creato in precedenza
    MPI_File_set_view(mat_file, 2 * sizeof(float), MPI_FLOAT, mat_darray, "native", MPI_INFO_NULL);

    MPI_File_read_all(mat_file, recv_block, submat_info->submat_row * submat_info->submat_col, MPI_FLOAT, &status);

    MPI_File_close(&mat_file);

    MPI_Type_free(&mat_darray);
    submat_info->submat = recv_block;
}

// Compute the process coordinates in the processg grid
void set_proc_grid_info(int pg_col, struct comm_info *comm_info)
{
    comm_info->pg_row_idx = comm_info->rank / pg_col;
    comm_info->pg_col_idx = comm_info->rank % pg_col;
}

// Calcolo delle informazioni sui blocchi per ogni processo
void compute_block_info(int row, int col, int row_block_size, int col_block_size, int pg_row, int pg_col, struct comm_info *comm_info, struct submat_info *submat_info)
{
    int submat_elem_per_row = 0, submat_elem_per_col = 0;
    int num_block_per_row_per_proc = 0, num_block_per_col_per_proc = 0;
    int num_extra_block_per_row = 0, num_extra_block_per_col = 0;
    int temp, rem_block_per_col, rem_block_per_row;

    /*I blocchi base sono intesi in numero di griglie complete e.g una matrice 16x7 divisa in blocchi 2x2 e process grid 2x2
     avrà solo un blocco completo per processo presente in ogni riga, quindi P00 avrà il blocco base 0 ed il blocco completo ma non base 2,
     mentre P01 avrà il blocco base 1 e il blocco incompleto 3 che avrà una sola colonna
    */
    num_block_per_row_per_proc = col / (col_block_size * pg_col); // Per ora sono solo blocchi base
    num_block_per_col_per_proc = row / (row_block_size * pg_row);

    // Calcolo dei blocchi extra per riga
    rem_block_per_row = col % col_block_size;
    temp = ((int)ceil((float)col / col_block_size)) % pg_col;

    if ((rem_block_per_row != 0) && (temp == 0))
        num_extra_block_per_row = pg_col;
    else
        num_extra_block_per_row = temp;

    // Calcolo dei blocchi extra per colonna
    rem_block_per_col = row % row_block_size;
    temp = ((int)ceil((float)row / row_block_size)) % pg_row;

    if (rem_block_per_col != 0 && (temp == 0))
        num_extra_block_per_col = pg_row;
    else
        num_extra_block_per_col = temp;

#if DEBUG_ELEMENT
    if (comm_info->rank == 0)
    {
        printf("DEBUG -> Number of base blocks per row per process: %d\n", num_block_per_row_per_proc);
        printf("DEBUG -> Number of base blocks per col per process: %d\n", num_block_per_col_per_proc);
        printf("DEBUG -> Number of extra blocks per row: %d\n", num_extra_block_per_row);
        printf("DEBUG -> Number of extra blocks per col: %d\n", num_extra_block_per_col);
    }
#endif

    // Assign extra block to first processes cyclically
    if (comm_info->pg_col_idx < num_extra_block_per_row)
    {
        num_block_per_row_per_proc++;
    }

    if (comm_info->pg_row_idx < num_extra_block_per_col)
    {
        num_block_per_col_per_proc++;
    }

    submat_elem_per_row = num_block_per_row_per_proc * col_block_size;
    submat_elem_per_col = num_block_per_col_per_proc * row_block_size;

    if ((comm_info->pg_col_idx == num_extra_block_per_row - 1) && (rem_block_per_row != 0))
    {
        submat_elem_per_row -= col_block_size - (rem_block_per_row);
    }

    if ((comm_info->pg_row_idx == num_extra_block_per_col - 1) && (rem_block_per_col != 0))
    {
        submat_elem_per_col -= row_block_size - (rem_block_per_col);
    }

#ifdef DEBUG_ELEMENT
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per row: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, num_block_per_row_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per col: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, num_block_per_col_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix row size: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_elem_per_row);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix col size: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_elem_per_col);
#endif

    submat_info->num_blocks_per_row = num_block_per_row_per_proc;
    submat_info->num_blocks_per_col = num_block_per_col_per_proc;
    submat_info->submat_row = submat_elem_per_col;
    submat_info->submat_col = submat_elem_per_row;
}

// Calcolo delle informazioni sui blocchi per ogni processo
void compute_row_block_info(int row, int col, int row_block_size, int pg_row, int pg_col, struct comm_info *comm_info, struct submat_info *submat_info)
{
    int submat_elem_per_row = 0, submat_elem_per_col = 0;
    int num_block_per_col_per_proc = 0;
    int num_extra_block_per_row = 0, num_extra_block_per_col = 0;
    int temp, rem_block_per_col, rem_block_per_row;

    struct comm_info *temp_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (temp_comm_info == NULL)
    {
        printf("Error in memory allocation for temp_comm_info in compute_block_info\n");
        exit(1);
    }

    temp_comm_info->rank = (comm_info->rank % pg_col); // TODO Vedere se farlo con comunicatore
    set_proc_grid_info(pg_col, temp_comm_info);

#if DEBUG_ELEMENT
    printf("\nRank %d in grid position (%d, %d) treated as rank %d in grid position (%d,%d)\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, temp_comm_info->rank, temp_comm_info->pg_row_idx, temp_comm_info->pg_col_idx);
#endif
    /*I blocchi base sono intesi in numero di griglie complete e.g una matrice 16x7 divisa in blocchi 2x2 e process grid 2x2
     avrà solo un blocco completo per processo presente in ogni riga, quindi P00 avrà il blocco base 0 ed il blocco completo ma non base 2,
     mentre P01 avrà il blocco base     MPI_Barrier(comm_info->comm);
        for(int i=0; i<submat_info->submat_row*submat_info->submat_col; i++){
            printf("Rank %d in grid (%d, %d) has element %f in pos %d of recv_block\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, recv_block[i], i);
        }1 e il blocco incompleto 3 che avrà una sola colonna
    */
    num_block_per_col_per_proc = row / (row_block_size * pg_col); // Per ora sono solo blocchi base

    // Calcolo dei blocchi extra per colonna, sulle righe non ce ne sono perche le prendo intere
    rem_block_per_col = row % row_block_size;
    temp = ((int)ceil((float)row / row_block_size)) % pg_col; // Metto %col perche le sto assegnando a colonne di processi

    if (rem_block_per_col != 0 && (temp == 0))
        num_extra_block_per_col = pg_col; // Metto %col perche le sto assegnando a colonne di processi
    else
        num_extra_block_per_col = temp;

#if DEBUG_ELEMENT
    if (comm_info->rank == 0)
    {
        printf("\nDEBUG -> Number of base blocks per row per process: %d\n", 1);
        printf("DEBUG -> Number of base blocks per col per process: %d\n", num_block_per_col_per_proc);
        printf("DEBUG -> Number of extra blocks per col: %d\n", num_extra_block_per_col);
    }
#endif

    // Stavolta devo assegnare ciclicamente i blocchi extra ai processi che nella process grid hanno indice di colonna minore di num_extra_block_per_col
    if (temp_comm_info->pg_col_idx < num_extra_block_per_col)
    {
        num_block_per_col_per_proc++;
    }

    submat_elem_per_col = num_block_per_col_per_proc * row_block_size;

    if ((temp_comm_info->pg_col_idx == num_extra_block_per_col - 1) && (rem_block_per_col != 0))
    {
        submat_elem_per_col -= row_block_size - (rem_block_per_col);
    }

#if DEBUG_ELEMENT
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per row: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, 1);
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per col: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, num_block_per_col_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix row size: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, col);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix col size: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_elem_per_col);
#endif

    submat_info->num_blocks_per_row = 1;
    submat_info->num_blocks_per_col = num_block_per_col_per_proc;
    submat_info->submat_row = submat_elem_per_col;
    submat_info->submat_col = col; // Full row in row block distribution

#ifdef DEBUG_ELEMENT
    printf("Rank %d in grid position (%d, %d) has %d x %d submat\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_info->submat_row, submat_info->submat_col);
#endif

    free(temp_comm_info);
}

// Distribuzione della matrice in blocchi di righe scon metodo block cyclic
void row_block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info)
{
    MPI_Status status;
    MPI_Datatype mat_darray;
    MPI_File mat_file;
    int dims[2] = {row, col};                                         // Dimensione matrice originale
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC}; // Metodo di distribuzione dei blocchi
    int dargs[2] = {block_size, col};                                 // Dimensione dei blocchi, voglio distribuire blocchi di righe intere quindi metto la size originale delle colonne
    int proc_dims[2] = {pg_col, 1};                                   // Dimensione della griglia di processi
    float *recv_block;

    recv_block = (float *)malloc(submat_info->submat_row * submat_info->submat_col * sizeof(float));
    if (recv_block == NULL)
    {
        printf("Error in memory allocation for recv_block in row_block_cyclic_distribution\n");
        exit(1);
    }

    /*
    Creazione del tipo di dato per la matrice distribuita, ogni processo vedrà solo la sua porzione di matrice,
    la porzione viene definita tramite block cyclic distribution
    */
    MPI_Type_create_darray(pg_col, (comm_info->rank) % pg_col, 2, dims, distribs, dargs, proc_dims, MPI_ORDER_C, MPI_FLOAT, &mat_darray);
    MPI_Type_commit(&mat_darray);

// Apertura collettiva del file
#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    if (comm_info->rank == 0)
    {
        printf("DEBUG -> Opening file %s\n", mat_path);
    }
#endif
    MPI_File_open(comm_info->comm, mat_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_file);
    if (mat_file == MPI_FILE_NULL)
    {
        printf("Error opening file %s in block cyclic distribution\n", mat_path);
        exit(1);
    }

    // Ogni processo ha una visione della matrice specificata dal darray creato in precedenza
    MPI_File_set_view(mat_file, 2 * sizeof(float), MPI_FLOAT, mat_darray, "native", MPI_INFO_NULL);

    MPI_File_read_all(mat_file, recv_block, submat_info->submat_row * submat_info->submat_col, MPI_FLOAT, &status);

    MPI_File_close(&mat_file);

    MPI_Type_free(&mat_darray);
    submat_info->submat = recv_block;
}

// Create communicator for row of processes
void create_row_comm(int pg_col, struct comm_info *comm_info, struct comm_info *row_comm_info)
{
    MPI_Comm row_comm;
    int row_rank, row_size;
    int color; // Ho al più pg_row colori
    color = comm_info->pg_row_idx;
    MPI_Comm_split(comm_info->comm, color, comm_info->rank, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    row_comm_info->comm = row_comm;
    row_comm_info->rank = row_rank;
    row_comm_info->size = row_size;
#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    printf("Rank %d in grid (%d, %d) has row color %d and row communicator rank %d and size %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, color, row_rank, row_size);
#endif
}

// Create communicator for col of processes
void create_col_comm(int pg_row, struct comm_info *comm_info, struct comm_info *col_comm_info)
{
    MPI_Comm col_comm;
    int col_rank, col_size;
    int color; // Ho al più pg_col colori
    color = comm_info->pg_col_idx % pg_row;
    MPI_Comm_split(comm_info->comm, color, comm_info->rank, &col_comm);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);
    col_comm_info->comm = col_comm;
    col_comm_info->rank = col_rank;
    col_comm_info->size = col_size;
#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    printf("Rank %d in grid (%d, %d) has column color %d and column communicator rank %d and size %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, color, col_rank, col_size);
#endif
}

// Create communicator for row of processes
void create_row_leader_comm(int pg_row, int pg_col, struct comm_info *comm_info, struct comm_info *row_leader_comm_info)
{
    int ranks_to_include[pg_row];
    MPI_Group group, row_leader_group;
    MPI_Comm row_leader_comm;
    int row_leader_comm_rank, row_leader_comm_size;
    

    for (int i = 0; i < pg_row; i++)
    {
        ranks_to_include[i] = i * pg_col;
    }

    MPI_Comm_group(comm_info->comm, &group);
    MPI_Group_incl(group, pg_row, ranks_to_include, &row_leader_group);
    MPI_Comm_create(comm_info->comm, row_leader_group, &row_leader_comm);

    if (row_leader_comm == MPI_COMM_NULL)
    {
#if defined(DEBUG) || defined(DEBUG_ELEMENT)
        printf("DEBUG -> Rank %d is not a row leader\n", comm_info->rank);
#endif
        row_leader_comm_info->comm = MPI_COMM_NULL;
        row_leader_comm_info->rank = MPI_UNDEFINED;
        return;
    }

    MPI_Comm_rank(row_leader_comm, &row_leader_comm_rank);
    MPI_Comm_size(row_leader_comm, &row_leader_comm_size);
    row_leader_comm_info->comm = row_leader_comm;
    row_leader_comm_info->rank = row_leader_comm_rank;
    row_leader_comm_info->size = row_leader_comm_size;
    row_leader_comm_info->pg_row_idx = comm_info->pg_col_idx;
    row_leader_comm_info->pg_col_idx = comm_info->pg_row_idx;

#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    if (row_leader_comm_info->rank != MPI_COMM_NULL)
        printf("DEBUG -> Rank %d in grid (%d, %d) belongs to row leader communicator of size %d with rank %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, row_leader_comm_size, row_leader_comm_rank);
#endif
}

void block_cyclic_write_result(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info)
{
    MPI_Status status;
    MPI_Datatype mat_darray;
    MPI_File mat_file;
    int dims[2] = {row, col};                                         // Dimensione matrice originale
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC}; // Metodo di distribuzione dei blocchi
    int dargs[2] = {block_size, col};                                 // Dimensione dei blocchi, voglio distribuire blocchi di righe intere quindi metto la size originale delle colonne
    int proc_dims[2] = {pg_col, 1};                                   // Dimensione della griglia di processi

    /*
    Creazione del tipo di dato per la matrice distribuita, ogni processo vedrà solo la sua porzione di matrice,
    la porzione viene definita tramite block cyclic distribution
    */
    MPI_Type_create_darray(pg_col, (comm_info->rank) % pg_col, 2, dims, distribs, dargs, proc_dims, MPI_ORDER_C, MPI_FLOAT, &mat_darray);
    MPI_Type_commit(&mat_darray);

// Apertura collettiva del file
#if defined(DEBUG) || defined(DEBUG_ELEMENT)
    if (comm_info->rank == 0)
    {
        printf("DEBUG -> Opening file %s\n", mat_path);
    }
#endif
    MPI_File_open(comm_info->comm, mat_path, MPI_MODE_WRONLY, MPI_INFO_NULL, &mat_file);
    if (mat_file == MPI_FILE_NULL)
    {
        printf("Error opening file %s in block cyclic distribution\n", mat_path);
        exit(1);
    }

    // Ogni processo ha una visione della matrice specificata dal darray creato in precedenza
    MPI_File_set_view(mat_file, 2 * sizeof(float), MPI_FLOAT, mat_darray, "native", MPI_INFO_NULL);

#ifdef DEBUG_ELEMENT
    printf("Rank %d in grid (%d, %d) writting %d x %d submat\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_info->submat_row, submat_info->submat_col);
#endif

    MPI_File_write_all(mat_file, submat_info->submat, submat_info->submat_row * submat_info->submat_col, MPI_FLOAT, &status);

    MPI_File_close(&mat_file);

    MPI_Type_free(&mat_darray);
}

void reset_matrix_c(char mat_c_path[128], char mat_c_path_check[128]){
    std::string c_path=mat_c_path;
    std::string c_path_check=mat_c_path_check;
    std::string cmd = "cp " + c_path_check + " " + c_path;
    system(cmd.c_str());
}

float *check_result(char mat_a_path[128], char mat_b_path[128], char mat_c_path[128], char mat_c_path_check[128], int r1, int c1, int c2)
{
    float reldiff = 0.0f;
    float diff = 0.0f;
    float *mat_a, *mat_b, *mat_c, *mat_c_check;
    float *res;
    MPI_File mat_a_file, mat_b_file, mat_c_file, mat_c_check_file;
    MPI_Status status;

    res=(float *)malloc(2*sizeof(float));
    if(res == NULL){
        printf("Error in memory allocation for res in check_result\n");
        exit(1);
    }

    mat_a = (float *)malloc(r1 * c1 * sizeof(float));
    if (mat_a == NULL)
    {
        printf("Error in memory allocation for matrix A\n");
        exit(1);
    }

    mat_b = (float *)malloc(c1 * c2 * sizeof(float));
    if (mat_b == NULL)
    {
        printf("Error in memory allocation for matrix B\n");
        exit(1);
    }

    mat_c = (float *)malloc(r1 * c2 * sizeof(float));
    if (mat_c == NULL)
    {
        printf("Error in memory allocation for matrix C\n");
        exit(1);
    }
    mat_c_check = (float *)malloc(r1 * c2 * sizeof(float));
    if (mat_c_check == NULL)
    {
        printf("Error in memory allocation for matrix C check\n");
        exit(1);
    }

    // Read matrix A
    MPI_File_open(MPI_COMM_SELF, mat_a_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_a_file);
    MPI_File_seek(mat_a_file, 2 * sizeof(int), MPI_SEEK_SET);
    MPI_File_read(mat_a_file, mat_a, r1 * c1, MPI_FLOAT, &status);

    // Read matrix B
    MPI_File_open(MPI_COMM_SELF, mat_b_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_b_file);
    MPI_File_seek(mat_b_file, 2 * sizeof(int), MPI_SEEK_SET);
    MPI_File_read(mat_b_file, mat_b, c1 * c2, MPI_FLOAT, &status);

    // Read matrix C
    MPI_File_open(MPI_COMM_SELF, mat_c_path, MPI_MODE_RDWR, MPI_INFO_NULL, &mat_c_file);
    MPI_File_seek(mat_c_file, 2 * sizeof(int), MPI_SEEK_SET);
    MPI_File_read(mat_c_file, mat_c, r1 * c2, MPI_FLOAT, &status);

    // Read matrix C check
    MPI_File_open(MPI_COMM_SELF, mat_c_path_check, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_c_check_file);
    MPI_File_seek(mat_c_check_file, 2 * sizeof(int), MPI_SEEK_SET);
    MPI_File_read(mat_c_check_file, mat_c_check, r1 * c2, MPI_FLOAT, &status);

    // Reset matrix C for repeatability before calculation
    reset_matrix_c(mat_c_path, mat_c_path_check);

#ifdef DEBUG_ELEMENT
    printf("Matrix A\n");
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c1; j++)
        {
            printf("%f ", mat_a[i * c1 + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B\n");
    for (int i = 0; i < c1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            printf("%f ", mat_b[i * c2 + j]);
        }
        printf("\n");
    }

    printf("\nMatrix C\n");
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            printf("%f ", mat_c[i * c2 + j]);
        }
        printf("\n");
    }

    printf("\nMatrix C check\n");
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            printf("%f ", mat_c_check[i * c2 + j]);
        }
        printf("\n");
    }
#endif

    int i, j, k;
    matrix_multiply(mat_a, mat_b, mat_c_check, r1, c1, c2, false);
//#ifdef DEBUG_ELEMENT
    printf("\nMatrix C check after computation\n");
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            printf("%f ", mat_c_check[i * c2 + j]);
        }
        printf("\n");
    }
//#endif
    int max_diff_i = -1;
    int max_diff_j = -1;
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            //printf("Mat_c_check[%d][%d] = %f\t", i, j, mat_c_check[i * c2 + j]);
            //printf("Mat_c[%d][%d] = %f\t", i, j, mat_c[i * c2 + j]);
            //printf("Diff = %f\n", abs(mat_c_check[i * c2 + j] - mat_c[i * c2 + j]));
            float maxabs = std::max(std::abs(mat_c_check[i * c2 + j]), std::abs(mat_c[i * c2 + j]));
            if (maxabs == 0.0)
                maxabs = 1.0;
            reldiff = std::max(reldiff, std::abs(mat_c_check[i * c2 + j] - mat_c[i * c2 + j]) / maxabs);
            diff = std::max(diff, std::abs(mat_c_check[i * c2 + j] - mat_c[i * c2 + j]));
            max_diff_i = i;
            max_diff_j = j;
        }
    }
    std::cout << "\tMax diff = " << diff << "\n\tMax rel diff = " << reldiff << std::endl;
    /*if ((i != -1) && (j != -1))
        printf("\tElement (%d,%d) caused maxdiff\n\tC[%d][%d] = %f\n\tC_check[%d][%d] = %f\n", max_diff_i, max_diff_j, max_diff_i, max_diff_j, mat_c[max_diff_i * c2 + max_diff_j], max_diff_i, max_diff_j, mat_c_check[max_diff_i * c2 + max_diff_j]);
    */
    res[0] = diff;
    res[1] = reldiff;
    return res;
}