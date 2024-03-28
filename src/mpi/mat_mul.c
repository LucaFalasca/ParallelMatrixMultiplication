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

struct proc_submatrix_info
{
    float *submatrix;                   // Pointer to the submatrix for this process
    int num_blocks_per_row;             // Number of blocks per row for this process
    int num_blocks_per_col;             // Number of blocks per column for this process
    int submatrix_row;                  // Number of rows of the submatrix for this process (aggregation of the blocks)
    int submatrix_col;                  // Number of columns of the submatrix for this process (aggregation of the blocks)
    struct block_info **ist_block_info; // Array of block info for this process blocks
} proc_submatrix_info;

struct proc_info
{
    int rank;                                    // Process rank
    int size;                                    // Number of processes
    MPI_Comm comm;                               // MPI communicator
    MPI_Comm row_comm;                           // MPI communicator for the row division
    MPI_Comm row_leader_comm;                    // MPI communicator for the row leader
    int row_leader_comm_rank;                         // Rank of the row leader
    int row_leader_comm_size;                         // Size of the row leader
    int row_comm_rank;                           // Rank in the row communicator
    int row_comm_size;                           // Size of the row communicator
    int pg_row_idx;                              // Process row index in the process grid
    int pg_col_idx;                              // Process column index in the process grid
    struct proc_submatrix_info *submat_A_info;   // Pointer to the submatrix info
    struct proc_submatrix_info *submat_B_info;   // Pointer to the submatrix info
    struct proc_submatrix_info *submat_res_info; // Pointer to the submatrix info
} proc_info;

void matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2);
bool seq_check_result(float *mat1, float *mat2, float *res, int r1, int c1, int c2);
void set_proc_grid_info(struct proc_info *proc_info, int pg_col);
struct proc_submatrix_info *compute_block_info(int row, int col, int row_block_size, int col_block_size, int pg_row, int pg_col, struct proc_info *proc_info);
struct proc_submatrix_info *compute_row_block_info(int row, int col, int row_block_size, int pg_row, int pg_col, struct proc_info *proc_info);
float *block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info);
float *row_block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info);
void create_row_comm(struct proc_info *proc_info, int pg_row);
void create_row_leader_comm(struct proc_info *proc_info, int pg_row);

int main(int argc, char *argv[])
{
    int row_a, col_a, row_b, col_b, pg_row, pg_col, block_size;
    float *submat_res, *submat_a, *submat_b, *partial_res;
    char mat_a_path[128], mat_b_path[128];
    struct proc_info *proc_info;
    MPI_Init(&argc, &argv);
    proc_info = (struct proc_info *)malloc(sizeof(struct proc_info));

    if (proc_info == NULL)
    {
        printf("Error in memory allocation for proc_info\n");
        exit(1);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &(proc_info->comm));
    MPI_Comm_rank(MPI_COMM_WORLD, &(proc_info->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(proc_info->size));

    /*Get parameters from cmd*/
    if (argc < 10)
    {
        printf("Usage ./a.out <nrproc> <ncproc> <blocks> <mat1path> <rows1> <cols1> <mat2path> <rows2> <cols2> \n");
        exit(1);
    }

    // Process grid size
    pg_row = atoi(argv[1]);
    pg_col = atoi(argv[2]);

    /*Check size compatibility for process grid*/
    if ((pg_row * pg_col) != proc_info->size)
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

    /*Check size compatibility for matrix multiply*/
    if (col_a != row_b)
    {
        printf("Incompatible matrix size for multiplication c1!=r2\n");
        exit(1);
    }

#ifdef DEBUG
    if (proc_info->rank == 0)
    {
        printf("DEBUG -> Number of processes: %d\n", proc_info->size);
        printf("DEBUG -> Process grid size: %d x %d\n", pg_row, pg_col);
        printf("DEBUG -> Block size: %d x %d\n", block_size, block_size);
        printf("DEBUG -> Matrix A path %s size: %d x %d\n", mat_a_path, row_a, col_a);
        printf("DEBUG -> Matrix B path %s size: %d x %d\n", mat_b_path, row_b, col_b);
    }
#endif
    // Each process calculates its position in the process grid
    set_proc_grid_info(proc_info, pg_col);

    /*Create row communicator, each row of processes get the same color as it contributes to the same row of the result matrix
    in this manner we build a leader-follower architecture to compute the effective row of the result matrix, each follower
    computes only only A*B and send the result to the leader that will sum the partial results and perform C+=A*B.
    All the followers work in different zones of the result matrix
    */
    create_row_comm(proc_info, pg_row);

    submat_a = block_cyclic_distribution(mat_a_path, row_a, col_a, block_size, pg_row, pg_col, proc_info);
    submat_b = row_block_cyclic_distribution(mat_b_path, row_b, col_b, block_size, pg_row, pg_col, proc_info);

    int submat_A_row = proc_info->submat_A_info->submatrix_row;
    int submat_A_col = proc_info->submat_A_info->submatrix_col;
    int submat_B_col = proc_info->submat_B_info->submatrix_col;

    // Allocate result submatrix TODO ognuno si alloca il suo pezzetto poi solo il processo leader si legge C e ci somma i parziali
    partial_res = (float *)malloc(submat_A_row * submat_B_col * sizeof(float));
    if (partial_res == NULL)
    {
        printf("Error in memory allocation for partial matrix result\n");
        exit(1);
    }
    printf("Rank %d in grid (%d, %d) has %dx%d submatrix of A and %dx%d submatrix of B and result submatrix of %dx%d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, submat_A_row, submat_A_col, proc_info->submat_B_info->submatrix_row, submat_B_col, submat_A_row, submat_B_col);

    // Perform multiplication of submatrix
    matrix_multiply(submat_a, submat_b, partial_res, submat_A_row, submat_A_col, submat_B_col);

    // Free submatrix a and b
    free(submat_a);
    free(submat_b);

    // Leader reduce of the partial results of all the followers
    if (proc_info->row_comm_rank == 0)
    {
        submat_res = (float *)malloc(submat_A_row * submat_B_col * sizeof(float));
        if (submat_res == NULL)
        {
            printf("Error in memory allocation for result submatrix\n");
            exit(1);
        }
    }
    MPI_Reduce(partial_res, submat_res, submat_A_row * submat_B_col, MPI_FLOAT, MPI_SUM, 0, proc_info->row_comm);

    // Free partial result matrix
    free(partial_res);

    //Create a communicator with only the row leaders which have to perform the MPI I/O ops on the result matrix file
    create_row_leader_comm(proc_info, pg_row);
    // TODO leader write result on correct position of the result matrix file
    // TODO leader read matrix file before performing the multiplication

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
            for (k = 0; k < c1; ++k)
            {
                res[i * c2 + j] = mat1[i * c1 + k] * mat2[k * c2 + j];
            }
        }
    }
}

bool seq_check_result(float *mat1, float *mat2, float *res, int r1, int c1, int c2)
{
    float *correct_res;
    correct_res = (float *)malloc(r1 * c2 * sizeof(float));
    if (correct_res == NULL)
    {
        printf("Error in memory allocation for check result\n");
        exit(1);
    }
    int i, j, k;
    for (i = 0; i < r1; i++)
    {
        for (j = 0; j < c2; j++)
        {
            correct_res[j * c2 + i] = 0;
            for (k = 0; k < c1; k++)
            {
                correct_res[j * c2 + i] += mat1[k * c1 + i] * mat2[j * c2 + k];
                if (correct_res[j * c2 + i] != res[j * c2 + i])
                {
                    printf("Error in position %d %d\n", i, j);
                    return false;
                }
            }
        }
    }
    return true;
}

// Distribuzione della matrice in blocchi con metodo block cyclic
float *block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info)
{
    MPI_Status status;
    MPI_Datatype mat_darray;
    MPI_File mat_file;
    int dims[2] = {row, col};                                         // Dimensione matrice originale
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC}; // Metodo di distribuzione dei blocchi
    int dargs[2] = {block_size, block_size};                          // Dimensione dei blocchi
    int proc_dims[2] = {pg_row, pg_col};                              // Dimensione della griglia di processi
    float *recv_block;

    proc_info->submat_A_info = compute_block_info(row, col, block_size, block_size, pg_row, pg_col, proc_info);

    recv_block = (float *)malloc(proc_info->submat_A_info->submatrix_row * proc_info->submat_A_info->submatrix_col * sizeof(float));
    recv_block = memset(recv_block, 0, proc_info->submat_A_info->submatrix_row * proc_info->submat_A_info->submatrix_col * sizeof(float));

    /*
    Creazione del tipo di dato per la matrice distribuita, ogni processo vedrà solo la sua porzione di matrice,
    la porzione viene definita tramite block cyclic distribution
    */
    MPI_Type_create_darray(proc_info->size, proc_info->rank, 2, dims, distribs, dargs, proc_dims, MPI_ORDER_C, MPI_FLOAT, &mat_darray);
    MPI_Type_commit(&mat_darray);

// Apertura collettiva del file
#ifdef DEBUG
    if (proc_info->rank == 0)
    {
        printf("DEBUG -> Opening file %s\n", mat_path);
    }
#endif
    MPI_File_open(proc_info->comm, mat_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_file);
    if (mat_file == MPI_FILE_NULL)
    {
        printf("Error opening file %s in block cyclic distribution\n", mat_path);
        exit(1);
    }

    // Ogni processo ha una visione della matrice specificata dal darray creato in precedenza
    MPI_File_set_view(mat_file, 2 * sizeof(float), MPI_FLOAT, mat_darray, "native", MPI_INFO_NULL);

    MPI_File_read_all(mat_file, recv_block, proc_info->submat_A_info->submatrix_row * proc_info->submat_A_info->submatrix_col, MPI_FLOAT, &status);

    MPI_File_close(&mat_file);

#ifdef DEBUG
    MPI_Barrier(proc_info->comm);
    for (int i = 0; i < (proc_info->submat_A_info->submatrix_row) * (proc_info->submat_A_info->submatrix_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submatrix of A: %f\n", proc_info->pg_row_idx, proc_info->pg_col_idx, recv_block[i]);
    }
#endif
    MPI_Type_free(&mat_darray);
    return recv_block;
}

// Compute the process coordinates in the processg grid
void set_proc_grid_info(struct proc_info *proc_info, int pg_col)
{
    proc_info->pg_row_idx = proc_info->rank / pg_col;
    proc_info->pg_col_idx = proc_info->rank % pg_col;
}

// Calcolo delle informazioni sui blocchi per ogni processo
struct proc_submatrix_info *compute_block_info(int row, int col, int row_block_size, int col_block_size, int pg_row, int pg_col, struct proc_info *proc_info)
{
    int submatrix_elem_per_row = 0, submatrix_elem_per_col = 0;
    int num_block_per_row_per_proc = 0, num_block_per_col_per_proc = 0;
    int num_extra_block_per_row = 0, num_extra_block_per_col = 0;
    int temp, rem_block_per_col, rem_block_per_row;

    struct proc_submatrix_info *submat_info = (struct proc_submatrix_info *)malloc(sizeof(struct proc_submatrix_info));
    if (submat_info == NULL)
    {
        printf("Error in memory allocation for proc_submatrix_info in compute_block_info\n");
        exit(1);
    }

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
    if (proc_info->rank == 0)
    {
        printf("DEBUG -> Number of base blocks per row per process: %d\n", num_block_per_row_per_proc);
        printf("DEBUG -> Number of base blocks per col per process: %d\n", num_block_per_col_per_proc);
        printf("DEBUG -> Number of extra blocks per row: %d\n", num_extra_block_per_row);
        printf("DEBUG -> Number of extra blocks per col: %d\n", num_extra_block_per_col);
    }
#endif

    // Assign extra block to first processes cyclically
    if (proc_info->pg_col_idx < num_extra_block_per_row)
    {
        num_block_per_row_per_proc++;
    }

    if (proc_info->pg_row_idx < num_extra_block_per_col)
    {
        num_block_per_col_per_proc++;
    }

    submatrix_elem_per_row = num_block_per_row_per_proc * col_block_size;
    submatrix_elem_per_col = num_block_per_col_per_proc * row_block_size;

    if ((proc_info->pg_col_idx == num_extra_block_per_row - 1) && (rem_block_per_row != 0))
    {
        submatrix_elem_per_row -= col_block_size - (rem_block_per_row);
    }

    if ((proc_info->pg_row_idx == num_extra_block_per_col - 1) && (rem_block_per_col != 0))
    {
        submatrix_elem_per_col -= row_block_size - (rem_block_per_col);
    }

#ifdef DEBUG
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per row: %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, num_block_per_row_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per col: %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, num_block_per_col_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix row size: %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, submatrix_elem_per_row);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix col size: %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, submatrix_elem_per_col);
#endif

    submat_info->num_blocks_per_row = num_block_per_row_per_proc;
    submat_info->num_blocks_per_col = num_block_per_col_per_proc;
    submat_info->submatrix_row = submatrix_elem_per_col;
    submat_info->submatrix_col = submatrix_elem_per_row;

    printf("Rank %d in grid position (%d, %d) has %d x %d submatrix of A\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, submat_info->submatrix_row, submat_info->submatrix_col);
    return submat_info;
}

// Calcolo delle informazioni sui blocchi per ogni processo
struct proc_submatrix_info *compute_row_block_info(int row, int col, int row_block_size, int pg_row, int pg_col, struct proc_info *proc_info)
{
    int submatrix_elem_per_row = 0, submatrix_elem_per_col = 0;
    int num_block_per_col_per_proc = 0;
    int num_extra_block_per_row = 0, num_extra_block_per_col = 0;
    int temp, rem_block_per_col, rem_block_per_row;

    struct proc_submatrix_info *submat_info = (struct proc_submatrix_info *)malloc(sizeof(struct proc_submatrix_info));
    if (submat_info == NULL)
    {
        printf("Error in memory allocation for proc_submatrix_info in compute_block_info\n");
        exit(1);
    }

    struct proc_info *temp_proc_info = (struct proc_info *)malloc(sizeof(struct proc_info));
    if (temp_proc_info == NULL)
    {
        printf("Error in memory allocation for temp_proc_info in compute_block_info\n");
        exit(1);
    }
    temp_proc_info->rank = (proc_info->rank % pg_col); // TODO Vedere se farlo con comunicatore
    set_proc_grid_info(temp_proc_info, pg_col);
#ifdef DEBUG
    printf("\nRank %d in grid position (%d, %d) treated as rank %d in grid position (%d,%d)\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, temp_proc_info->rank, temp_proc_info->pg_row_idx, temp_proc_info->pg_col_idx);
#endif
    /*I blocchi base sono intesi in numero di griglie complete e.g una matrice 16x7 divisa in blocchi 2x2 e process grid 2x2
     avrà solo un blocco completo per processo presente in ogni riga, quindi P00 avrà il blocco base 0 ed il blocco completo ma non base 2,
     mentre P01 avrà il blocco base 1 e il blocco incompleto 3 che avrà una sola colonna
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
    if (proc_info->rank == 0)
    {
        printf("\nDEBUG -> Number of base blocks per row per process: %d\n", 1);
        printf("DEBUG -> Number of base blocks per col per process: %d\n", num_block_per_col_per_proc);
        printf("DEBUG -> Number of extra blocks per col: %d\n", num_extra_block_per_col);
    }
#endif

    // Stavolta devo assegnare ciclicamente i blocchi extra ai processi che nella process grid hanno indice di colonna minore di num_extra_block_per_col
    if (temp_proc_info->pg_col_idx < num_extra_block_per_col)
    {
        num_block_per_col_per_proc++;
    }

    submatrix_elem_per_col = num_block_per_col_per_proc * row_block_size;

    if ((temp_proc_info->pg_col_idx == num_extra_block_per_col - 1) && (rem_block_per_col != 0))
    {
        submatrix_elem_per_col -= row_block_size - (rem_block_per_col);
    }

#ifdef DEBUG
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per row: %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, 1);
    printf("DEBUG -> Rank %d pos (%d,%d) Number of blocks per col: %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, num_block_per_col_per_proc);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix row size: %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, col);
    printf("DEBUG -> Rank %d pos (%d,%d) Submatrix col size: %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, submatrix_elem_per_col);
#endif

    submat_info->num_blocks_per_row = 1;
    submat_info->num_blocks_per_col = num_block_per_col_per_proc;
    submat_info->submatrix_row = submatrix_elem_per_col;
    submat_info->submatrix_col = col; // Full row in row block distribution

    printf("Rank %d in grid position (%d, %d) has %d x %d submatrix of B\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, submat_info->submatrix_row, submat_info->submatrix_col);

    free(temp_proc_info);
    return submat_info;
}

float *row_block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info)
{
    MPI_Status status;
    MPI_Datatype mat_darray;
    MPI_File mat_file;
    int dims[2] = {row, col};                                         // Dimensione matrice originale
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC}; // Metodo di distribuzione dei blocchi
    int dargs[2] = {block_size, col};                                 // Dimensione dei blocchi, voglio distribuire blocchi di righe intere quindi metto la size originale delle colonne
    int proc_dims[2] = {pg_col, 1};                                   // Dimensione della griglia di processi
    float *recv_block;

    proc_info->submat_B_info = compute_row_block_info(row, col, block_size, pg_row, pg_col, proc_info);

    recv_block = (float *)malloc(proc_info->submat_B_info->submatrix_row * proc_info->submat_B_info->submatrix_col * sizeof(float));
    recv_block = memset(recv_block, 0, proc_info->submat_B_info->submatrix_row * proc_info->submat_B_info->submatrix_col * sizeof(float));

    /*
    Creazione del tipo di dato per la matrice distribuita, ogni processo vedrà solo la sua porzione di matrice,
    la porzione viene definita tramite block cyclic distribution
    */
    MPI_Type_create_darray(pg_col, (proc_info->rank) % pg_col, 2, dims, distribs, dargs, proc_dims, MPI_ORDER_C, MPI_FLOAT, &mat_darray);
    MPI_Type_commit(&mat_darray);

// Apertura collettiva del file
#ifdef DEBUG
    if (proc_info->rank == 0)
    {
        printf("DEBUG -> Opening file %s\n", mat_path);
    }
#endif
    MPI_File_open(proc_info->comm, mat_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_file);
    if (mat_file == MPI_FILE_NULL)
    {
        printf("Error opening file %s in block cyclic distribution\n", mat_path);
        exit(1);
    }

    // Ogni processo ha una visione della matrice specificata dal darray creato in precedenza
    MPI_File_set_view(mat_file, 2 * sizeof(float), MPI_FLOAT, mat_darray, "native", MPI_INFO_NULL);

    MPI_File_read_all(mat_file, recv_block, proc_info->submat_B_info->submatrix_row * proc_info->submat_B_info->submatrix_col, MPI_FLOAT, &status);

    MPI_File_close(&mat_file);

#ifdef DEBUG
    MPI_Barrier(proc_info->comm);
    for (int i = 0; i < (proc_info->submat_B_info->submatrix_row) * (proc_info->submat_B_info->submatrix_col); i++)
    {
        printf("DEBUG -> Rank (%d, %d) submatrix of B: %f\n", proc_info->pg_row_idx, proc_info->pg_col_idx, recv_block[i]);
    }
#endif
    MPI_Type_free(&mat_darray);
    return recv_block;
}

// Create communicator for row of processes
void create_row_comm(struct proc_info *proc_info, int pg_row)
{
    MPI_Comm row_comm;
    int row_rank, row_size;
    int color; // Ho al più pg_row colori
    color = proc_info->pg_row_idx % pg_row;
    MPI_Comm_split(proc_info->comm, color, proc_info->rank, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    proc_info->row_comm = row_comm;
    proc_info->row_comm_rank = row_rank;
    proc_info->row_comm_size = row_size;
#ifdef DEBUG
    printf("Rank %d in grid (%d, %d) has color %d and row communicator rank %d and size %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, color, row_rank, row_size);
#endif
}

void create_row_leader_comm(struct proc_info *proc_info, int pg_row)
{   
    int ranks_to_include[pg_row]; // Add only the current process
    MPI_Group group, row_leader_group;
    MPI_Comm row_leader_comm;
    int row_leader_comm_rank, row_leader_comm_size;

    for(int i=0; i<pg_row; i++){
        ranks_to_include[i] = i*pg_row;
    }
    
    MPI_Comm_group(proc_info->comm, &group);
    MPI_Group_incl(group, pg_row, ranks_to_include, &row_leader_group);
    MPI_Comm_create(proc_info->comm, row_leader_group, &row_leader_comm);

    if(row_leader_comm == MPI_COMM_NULL){
        proc_info->row_leader_comm_rank = MPI_UNDEFINED;
        return;
    }

    MPI_Comm_rank(row_leader_comm, &row_leader_comm_rank);
    MPI_Comm_size(row_leader_comm, &row_leader_comm_size);
    proc_info->row_leader_comm = row_leader_comm;
    proc_info->row_leader_comm_rank = row_leader_comm_rank;
    proc_info->row_leader_comm_size = row_leader_comm_size;

#ifdef DEBUG
    if(proc_info->rank != MPI_UNDEFINED)
        printf("Rank %d in grid (%d, %d) belongs to row leader communicator of size %d with rank %d\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, row_leader_comm_size, row_leader_comm_rank);
#endif
}
