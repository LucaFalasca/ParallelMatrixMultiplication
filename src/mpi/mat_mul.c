#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "util/utils.h"
#define BLOCK_ROWS 2
#define BLOCK_COLS 2
#define RSRC 0
#define CSRC 0

struct block_info
{
    int num_row;        // Number of rows in the block
    int num_col;        // Number of columns in the block
    int row_start_gidx; // Global index of the first row in the block
    int col_start_gidx; // Global index of the first column in the block
} block_info;

struct submat_info
{
    float *submat;                      // Pointer to the submat for this process
    int num_blocks_per_row;             // Number of blocks per row for this process
    int num_blocks_per_col;             // Number of blocks per column for this process
    int submat_row;                     // Number of rows of the submat for this process (aggregation of the blocks)
    int submat_col;                     // Number of columns of the submat for this process (aggregation of the blocks)
    struct block_info **ist_block_info; // Array of block info for this process blocks
} submat_info;

struct comm_info
{
    MPI_Comm comm;  // MPI communicator
    int rank;       // Rank of the process in the communicator
    int size;       // Number of processes in the communicator
    int pg_row_idx; // Process row index in the process grid
    int pg_col_idx; // Process column index in the process grid
} comm_info;

void matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2);
bool seq_check_result(char mat_a_path[128], char mat_b_path[128], char mat_c_path[128], char mat_c_path_check[128], int r1, int c1, int c2);
void set_proc_grid_info(int pg_col, struct comm_info *comm_info);
void compute_block_info(int row, int col, int row_block_size, int col_block_size, int pg_row, int pg_col, struct comm_info *comm_info, struct submat_info *submat_info);
void compute_row_block_info(int row, int col, int row_block_size, int pg_row, int pg_col, struct comm_info *comm_info, struct submat_info *submat_info, bool isC);
void block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info);
void row_block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info, bool isC);
void create_row_comm(int pg_row, struct comm_info *comm_info, struct comm_info *row_comm_info);
void create_row_leader_comm(int pg_row, struct comm_info *comm_info, struct comm_info *row_leader_comm_info);
void block_cyclic_write_result(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int row_a, col_a, row_b, col_b, pg_row, pg_col, block_size;
    float *partial_res;
    char mat_a_path[128], mat_b_path[128], mat_c_path[128], mat_c_path_check[128];
    struct submat_info *submat_A_info, *submat_B_info, *submat_C_info;
    struct comm_info *comm_info, *row_comm_info, *row_leader_comm_info;

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
    row_leader_comm_info = (struct comm_info *)malloc(sizeof(struct comm_info));
    if (row_leader_comm_info == NULL)
    {
        printf("Error in memory allocation for row_leader_comm_info\n");
        exit(1);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &(comm_info->comm));
    MPI_Comm_rank(MPI_COMM_WORLD, &(comm_info->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(comm_info->size));

    /*Get parameters from cmd*/
    if (argc < 12)
    {
        printf("Usage ./a.out <nrproc> <ncproc> <blocks> <matApath> <rowsA> <colsA> <matBpath> <rowsB> <colsB> <matCpath> <matCpath_check\n");
        exit(1);
    }

    // Process grid size
    pg_row = atoi(argv[1]);
    pg_col = atoi(argv[2]);

    /*Check size compatibility for process grid*/
    if ((pg_row * pg_col) != comm_info->size)
    {
        printf("Process grid size incompatible with number of processes spawned\n");
        exit(1);
    }

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
    if (comm_info->rank == 0)
    {
        printf("DEBUG -> Number of processes: %d\n", comm_info->size);
        printf("DEBUG -> Process grid size: %d x %d\n", pg_row, pg_col);
        printf("DEBUG -> Block size: %d x %d\n", block_size, block_size);
        printf("DEBUG -> Matrix A path %s size: %d x %d\n", mat_a_path, row_a, col_a);
        printf("DEBUG -> Matrix B path %s size: %d x %d\n", mat_b_path, row_b, col_b);
        printf("DEBUG -> Matrix C path %s size: %d x %d\n", mat_c_path, row_a, col_b);
        printf("DEBUG -> Matrix C path check %s size: %d x %d\n", mat_c_path_check, row_a, col_b);
    }
    // Each process calculates its position in the process grid
    set_proc_grid_info(pg_col, comm_info);

    /*Create row communicator, each row of processes get the same color as it contributes to the same row of the result matrix
    in this manner we build a leader-follower architecture to compute the effective row of the result matrix, each follower
    computes only only A*B and send the result to the leader that will sum the partial results and perform C+=A*B.
    All the followers work in different zones of the result matrix
    */
    create_row_comm(pg_row, comm_info, row_comm_info);

    // Create a communicator with only the row leaders which have to perform the MPI I/O ops on the result matrix file
    create_row_leader_comm(pg_row, comm_info, row_leader_comm_info);

    block_cyclic_distribution(mat_a_path, row_a, col_a, block_size, pg_row, pg_col, submat_A_info, comm_info);
    row_block_cyclic_distribution(mat_b_path, row_b, col_b, block_size, pg_row, pg_col, submat_B_info, comm_info, false);

    int submat_A_row = submat_A_info->submat_row;
    int submat_A_col = submat_A_info->submat_col;
    int submat_B_col = submat_B_info->submat_col;

    // Only the process leader of the row will have the result submat
    if (row_leader_comm_info->comm != MPI_COMM_NULL)
    {
        /*submat_C_info->submat = (float *)malloc(submat_A_row * submat_B_col * sizeof(float));
        if (submat_C_info->submat == NULL)
        {
            printf("Error in memory allocation for matrix C submat\n");
            exit(1);
        }
        memset(submat_C_info->submat, 0, submat_A_row * submat_B_col * sizeof(float));
        submat_C_info->submat_row = submat_A_row;
        submat_C_info->submat_col = submat_B_col;*/
        // Block cyclic distribution of C
        row_block_cyclic_distribution(mat_c_path, row_a, col_b, block_size, 1, pg_col, submat_C_info, row_leader_comm_info, true);
#ifdef DEBUG
        MPI_Barrier(row_leader_comm_info->comm);
        for (int i = 0; i < submat_C_info->submat_row * submat_C_info->submat_col; i++)
        {
            printf("Rank %d in grid (%d, %d) has element %f in pos %d of submat of C\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_C_info->submat[i], i);
        }
#endif
    }

    // Allocate partial result submatrix
    partial_res = (float *)malloc(submat_A_row * submat_B_col * sizeof(float));
    if (partial_res == NULL)
    {
        printf("Error in memory allocation for partial matrix result\n");
        exit(1);
    }

    printf("Rank %d in grid (%d, %d) has %dx%d submat of A, %dx%d submat of B, %dx%d submat of C and %dx%d partial result\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_A_row, submat_A_col, submat_B_info->submat_row, submat_B_col, submat_C_info->submat_row, submat_C_info->submat_col, submat_A_row, submat_B_col);

    // Perform multiplication of submat
    matrix_multiply(submat_A_info->submat, submat_B_info->submat, partial_res, submat_A_row, submat_A_col, submat_B_col);

    // Free submat a and b
    free(submat_A_info);
    free(submat_B_info);

    MPI_Reduce(partial_res, submat_C_info->submat, submat_A_row * submat_B_col, MPI_FLOAT, MPI_SUM, 0, row_comm_info->comm);

    // Free partial result matrix
    free(partial_res);

    // Leader write result
    if (row_leader_comm_info->comm != MPI_COMM_NULL)
    {  
        block_cyclic_write_result(mat_c_path, row_a, col_b, block_size, 1, pg_col, submat_C_info, row_leader_comm_info);
    }

    // Free submat C
    free(submat_C_info);

    MPI_Barrier(comm_info->comm);
    if(comm_info->rank == 0){
        printf("\n\n\n\nRank 0 checking result...\n");
        bool check=seq_check_result(mat_a_path, mat_b_path, mat_c_path, mat_c_path_check, row_a, col_a, col_b);
        if(check)
            printf("Result check passed\n");
        else
            printf("Result check failed\n");
    }
    

    MPI_Finalize();
    return 0;
}

void matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2)
{
    int i, j, k;
    for (i = 0; i < r1; ++i)
    {
        for (j = 0; j < c2; ++j)
        {   
            res[i * c2 + j] = 0;
            for (k = 0; k < c1; ++k)
            {
                res[i * c2 + j] += mat1[i * c1 + k] * mat2[k * c2 + j];
            }
        }
    }
}

bool seq_check_result(char mat_a_path[128], char mat_b_path[128], char mat_c_path[128], char mat_c_path_check[128], int r1, int c1, int c2)
{
    float *mat_a, *mat_b, *mat_c, *mat_c_check;
    MPI_File mat_a_file, mat_b_file, mat_c_file, mat_c_check_file;
    MPI_Status status;

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
    MPI_File_seek(mat_a_file, 2*sizeof(int), MPI_SEEK_SET);
    MPI_File_read(mat_a_file, mat_a, r1*c1, MPI_FLOAT, &status);

    // Read matrix B
    MPI_File_open(MPI_COMM_SELF, mat_b_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_b_file);
    MPI_File_seek(mat_b_file, 2*sizeof(int), MPI_SEEK_SET);
    MPI_File_read(mat_b_file, mat_b, c1*c2, MPI_FLOAT, &status);

    // Read matrix C
    MPI_File_open(MPI_COMM_SELF, mat_c_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_c_file);
    MPI_File_seek(mat_c_file, 2*sizeof(int), MPI_SEEK_SET);
    MPI_File_read(mat_c_file, mat_c, r1*c2, MPI_FLOAT, &status);

    // Read matrix C check
    MPI_File_open(MPI_COMM_SELF, mat_c_path_check, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_c_check_file);
    MPI_File_seek(mat_c_check_file, 2*sizeof(int), MPI_SEEK_SET);
    MPI_File_read(mat_c_check_file, mat_c_check, r1*c2, MPI_FLOAT, &status);

    printf("Matrix A\n");
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c1; j++)
        {
            printf("%f ", mat_a[i * c1 + j]);
        }
        printf("\n");
    }

    printf("Matrix B\n");
    for (int i = 0; i < c1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            printf("%f ", mat_b[i * c2 + j]);
        }
        printf("\n");
    }

    printf("Matrix C\n");
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            printf("%f ", mat_c[i * c2 + j]);
        }
        printf("\n");
    }

    printf("Matrix C check\n");
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            printf("%f ", mat_c_check[i * c2 + j]);
        }
        printf("\n");
    }

    int i, j, k;
    matrix_multiply(mat_a, mat_b, mat_c_check, r1, c1, c2);
    for (i = 0; i < r1; i++)
    {
        for (j = 0; j < c2; j++)
        {
            if(mat_c[i*c2+j]!=mat_c_check[i*c2+j]){
                printf("Error in position %d %d\n", i, j);
                //Restore matrix C for repeatability
                MPI_File_write(mat_c_file, mat_c_check, r1*c2, MPI_FLOAT, &status);
                return false;
            }
            //printf("%f ", mat_c_check[i * c2 + j]);
        }
        printf("\n");
    }

    //Restore matrix C for repeatability
    MPI_File_write(mat_c_file, mat_c_check, r1*c2, MPI_FLOAT, &status);
    return true;
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

    compute_block_info(row, col, block_size, block_size, pg_row, pg_col, comm_info, submat_info);

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
#ifdef DEBUG
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

#ifdef DEBUG
    MPI_Barrier(comm_info->comm);
    for (int i = 0; i < (submat_info->submat_row) * (submat_info->submat_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submat of A: %f\n", comm_info->pg_row_idx, comm_info->pg_col_idx, recv_block[i]);
    }
#endif
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

#ifdef DEBUG
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

#ifdef DEBUG
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per row: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, num_block_per_row_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per col: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, num_block_per_col_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix row size: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_elem_per_row);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix col size: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_elem_per_col);
#endif

    submat_info->num_blocks_per_row = num_block_per_row_per_proc;
    submat_info->num_blocks_per_col = num_block_per_col_per_proc;
    submat_info->submat_row = submat_elem_per_col;
    submat_info->submat_col = submat_elem_per_row;
#ifdef DEBUG
    printf("Rank %d in grid position (%d, %d) has %d x %d submat of A\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_info->submat_row, submat_info->submat_col);
#endif
}

// Calcolo delle informazioni sui blocchi per ogni processo
void compute_row_block_info(int row, int col, int row_block_size, int pg_row, int pg_col, struct comm_info *comm_info, struct submat_info *submat_info, bool isC)
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
#ifdef DEBUG
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

#ifdef DEBUG
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

#ifdef DEBUG
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per row: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, 1);
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per col: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, num_block_per_col_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix row size: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, col);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix col size: %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_elem_per_col);
#endif

    submat_info->num_blocks_per_row = 1;
    submat_info->num_blocks_per_col = num_block_per_col_per_proc;
    submat_info->submat_row = submat_elem_per_col;
    submat_info->submat_col = col; // Full row in row block distribution

#ifdef DEBUG
    // TODO togliere è per debug
    if (isC)
        printf("Rank %d in grid position (%d, %d) has %d x %d submat of C\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_info->submat_row, submat_info->submat_col);
    else
        printf("Rank %d in grid position (%d, %d) has %d x %d submat of B\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_info->submat_row, submat_info->submat_col);
#endif

    free(temp_comm_info);
}

// Distribuzione della matrice in blocchi di righe scon metodo block cyclic
void row_block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info, bool isC)
{
    MPI_Status status;
    MPI_Datatype mat_darray;
    MPI_File mat_file;
    int dims[2] = {row, col};                                         // Dimensione matrice originale
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC}; // Metodo di distribuzione dei blocchi
    int dargs[2] = {block_size, col};                                 // Dimensione dei blocchi, voglio distribuire blocchi di righe intere quindi metto la size originale delle colonne
    int proc_dims[2] = {pg_col, 1};                                   // Dimensione della griglia di processi
    float *recv_block;
    compute_row_block_info(row, col, block_size, pg_row, pg_col, comm_info, submat_info, isC);

    recv_block = (float *)malloc(submat_info->submat_row * submat_info->submat_col * sizeof(float));
    if (recv_block == NULL)
    {
        printf("Error in memory allocation for recv_block in row_block_cyclic_distribution\n");
        exit(1);
    }
    // recv_block = memset(recv_block, 0, submat_info->submat_row * submat_info->submat_col * sizeof(float));

    /*
    Creazione del tipo di dato per la matrice distribuita, ogni processo vedrà solo la sua porzione di matrice,
    la porzione viene definita tramite block cyclic distribution
    */
    MPI_Type_create_darray(pg_col, (comm_info->rank) % pg_col, 2, dims, distribs, dargs, proc_dims, MPI_ORDER_C, MPI_FLOAT, &mat_darray);
    MPI_Type_commit(&mat_darray);

// Apertura collettiva del file
#ifdef DEBUG
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

#ifdef DEBUG
    MPI_Barrier(comm_info->comm);
    for (int i = 0; i < (submat_info->submat_row) * (submat_info->submat_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submat of B: %f\n", comm_info->pg_row_idx, comm_info->pg_col_idx, recv_block[i]);
    }
#endif
    MPI_Type_free(&mat_darray);
    submat_info->submat = recv_block;
}

// Create communicator for row of processes
void create_row_comm(int pg_row, struct comm_info *comm_info, struct comm_info *row_comm_info)
{
    MPI_Comm row_comm;
    int row_rank, row_size;
    int color; // Ho al più pg_row colori
    color = comm_info->pg_row_idx % pg_row;
    MPI_Comm_split(comm_info->comm, color, comm_info->rank, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    row_comm_info->comm = row_comm;
    row_comm_info->rank = row_rank;
    row_comm_info->size = row_size;
#ifdef DEBUG
    printf("Rank %d in grid (%d, %d) has color %d and row communicator rank %d and size %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, color, row_rank, row_size);
#endif
}

// Create communicator for row of processes
void create_row_leader_comm(int pg_row, struct comm_info *comm_info, struct comm_info *row_leader_comm_info)
{
    int ranks_to_include[pg_row]; // Add only the current process
    MPI_Group group, row_leader_group;
    MPI_Comm row_leader_comm;
    int row_leader_comm_rank, row_leader_comm_size;

    for (int i = 0; i < pg_row; i++)
    {
        ranks_to_include[i] = i * pg_row;
    }

    MPI_Comm_group(comm_info->comm, &group);
    MPI_Group_incl(group, pg_row, ranks_to_include, &row_leader_group);
    MPI_Comm_create(comm_info->comm, row_leader_group, &row_leader_comm);

    if (row_leader_comm == MPI_COMM_NULL)
    {
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

#ifdef DEBUG
    if (comm_info->rank != MPI_UNDEFINED)
        printf("Rank %d in grid (%d, %d) belongs to row leader communicator of size %d with rank %d\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, row_leader_comm_size, row_leader_comm_rank);
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
#ifdef DEBUG
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

    //Print what is going to be written
    for (int i = 0; i < submat_info->submat_row * submat_info->submat_col; i++)
    {
        printf("Rank %d in grid (%d, %d) has element %f in pos %d of submat of C to write\n", comm_info->rank, comm_info->pg_row_idx, comm_info->pg_col_idx, submat_info->submat[i], i);
    }

    MPI_File_write_all(mat_file, submat_info->submat, submat_info->submat_row * submat_info->submat_col, MPI_FLOAT, &status);

    MPI_File_close(&mat_file);

    MPI_Type_free(&mat_darray);
}