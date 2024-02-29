#include<mpi.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<stdbool.h>
#include <math.h>
#include "util/utils.h"
#define BLOCK_ROWS 2 
#define BLOCK_COLS 2
#define RSRC 0
#define CSRC 0
#define DATA_REP "native"

struct block_info{
    int num_row; //Number of rows in the block
    int num_col; //Number of columns in the block
    int row_start_gidx; //Global index of the first row in the block
    int col_start_gidx; //Global index of the first column in the block
    struct block_info *next; //Pointer to the next block info
} block_info;

struct proc_submatrix_info{
    float *submatrix; //Pointer to the submatrix for this process
    int num_blocks; //Number of blocks for this process
    int submatrix_row; //Number of rows of the submatrix for this process (aggregation of the blocks)
    int submatrix_col; //Number of columns of the submatrix for this process (aggregation of the blocks)
    struct block_info **list_block_info; //List of block info for this process blocks
} proc_submatrix_info;

struct proc_info{
    int rank; //Process rank
    int size; //Number of processes
    MPI_Comm comm; //MPI communicator
    int pg_row_idx; //Process row index in the process grid 
    int pg_col_idx; //Process column index in the process grid
    int pg_row; //Number of rows in the process grid
    int pg_col; //Number of columns in the process grid
    struct proc_submatrix_info *submat_info; //Pointer to the submatrix info
} proc_info;

void matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2);
bool seq_check_result(float *mat1, float *mat2, float *res, int r1, int c1, int c2);
void row_block_cyclic_distribution_old(float *mat, int row, int col, int block_size, int npc, MPI_Comm comm, int rank);
void measure_submatrix_size(int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info);
void block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info);
void set_proc_grid_info(struct proc_info* proc_info, int pg_col);

int main(int argc, char *argv[]){
    int row_a, col_a, row_b, col_b, pg_row, pg_col, block_size;
    float *mat_res;
    char mat_a_path[128], mat_b_path[128];
    struct proc_info *proc_info;
    MPI_Init(&argc, &argv);
    proc_info = (struct proc_info *) malloc(sizeof(struct proc_info));

    if(proc_info==NULL){
        printf("Error in memory allocation for proc_info\n");
        exit(1);
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &(proc_info->comm));
    MPI_Comm_rank(MPI_COMM_WORLD, &(proc_info->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(proc_info->size));
    
    /*Get parameters from cmd*/
    if(argc<10){
        printf("Usage ./a.out <nrproc> <ncproc> <blocks> <mat1path> <rows1> <cols1> <mat2path> <rows2> <cols2> \n");
        exit(1);
    }

    //Process grid size
    pg_row = atoi(argv[1]);
    pg_col = atoi(argv[2]);

    /*Check size compatibility for process grid*/
    if((pg_row*pg_col)!=proc_info->size){
        printf("Process grid size incompatible with number of processes spawned\n");
        exit(1);
    }

    //Block size
    block_size = atoi(argv[3]);

    //Matrix A data
    strcpy(mat_a_path, argv[4]);
    row_a = atoi(argv[5]);
    col_a = atoi(argv[6]);

    //Matrix B data
    strcpy(mat_b_path, argv[7]);
    row_b = atoi(argv[8]);
    col_b = atoi(argv[9]);

    /*Check size compatibility for matrix multiply*/
    if(col_a!=row_b){
        printf("Incompatible matrix size for multiplication c1!=r2\n");
        exit(1);
    }

    #ifdef DEBUG
        if(rank==0){
            printf("Number of processes: %d\n", proc_info->size);
            printf("Process grid size: %d x %d\n", pg_row, pg_col);
            printf("Block size: %d x %d\n", block_size, block_size);
            printf("Matrix A path %s size: %d x %d\n", mat_a_path, row_a, col_a);
            printf("Matrix B path %s size: %d x %d\n", mat_b_path, row_b, col_b);
        }
    #endif
    
    block_cyclic_distribution(mat_a_path, row_a, col_a, block_size, pg_row, pg_col, proc_info);
    MPI_Finalize();
    return 0;
}

void matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2){
    int i, j, k;
    for (i = 0; i < r1; i++) {
        for (j = 0; j < c2; j++) {
            res[j*c2+i] = 0;
            for (k = 0; k < c1; k++) {
                res[i*c2+j] += mat1[i*c1+k] * mat2[k*c2+j];
            }
        }
    }
}

bool seq_check_result(float *mat1, float *mat2, float *res, int r1, int c1, int c2){
    float *correct_res;
    correct_res = (float *) malloc(r1*c2*sizeof(float));
    if(correct_res==NULL){
        printf("Error in memory allocation for check result\n");
        exit(1);
    }
    int i, j, k;
    for (i = 0; i < r1; i++) {
        for (j = 0; j < c2; j++) {
            correct_res[j*c2+i] = 0;
            for (k = 0; k < c1; k++) {
                correct_res[j*c2+i] += mat1[k*c1+i] * mat2[j*c2+k];
                if(correct_res[j*c2+i]!=res[j*c2+i]){
                    printf("Error in position %d %d\n", i, j);
                    return false;
                }
            }
        }
    }
    return true;
}

//TODO
void block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info){
    MPI_Status status;
    MPI_Datatype mat_darray;
    MPI_File mat_file;
    int dims[2] = {row, col}; //Dimensione matrice originale
    int distribs[2] = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC}; //Metodo di distribuzione dei blocchi
    int dargs[2] = {block_size, block_size}; //Dimensione dei blocchi
    int proc_dims[2] = {pg_row, pg_col}; //Dimensione della griglia di processi
    float *recv_block;

    set_proc_grid_info(proc_info, pg_col);
    measure_submatrix_size(row, col, block_size, proc_info);


    /*
    Creazione del tipo di dato per la matrice distribuita, ogni processo vedrà solo la sua porzione di matrice,
    la porzione viene definita tramite block cyclic distribution
    */
    MPI_Type_create_darray(proc_info->size, proc_info->rank, 2, dims, distribs,dargs, proc_dims, MPI_ORDER_C, MPI_INT, &mat_darray);
    MPI_Type_commit(&mat_darray);

    //Apertura collettiva del file
    MPI_File_open(proc_info->comm, mat_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_file);
    if (mat_file == MPI_FILE_NULL) {
	    printf("File open error in block cyclic distribution\n");
        exit(1);
    }

    //Ogni processo ha una visione della matrice specificata dal darray creato in precedenza
    MPI_File_set_view(mat_file, 2*sizeof(int), MPI_FLOAT, mat_darray, DATA_REP, MPI_INFO_NULL);
    
    //MPI_File_read_all(mat_file, recv_block, sub_matrix_row * sub_matrix_col, MPI_FLOAT, &status);

    MPI_File_close(&mat_file);
    //MPI_Type_free(&mat_darray);
    
}

//Compute the process coordinates in the processg grid
void set_proc_grid_info(struct proc_info* proc_info, int pg_col){
    proc_info->pg_row_idx = proc_info->rank/pg_col;
    proc_info->pg_col_idx = proc_info->rank%pg_col;
}

void measure_submatrix_size(int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info){
    int num_blocks_row=ceil(row/block_size);
    int num_blocks_col=ceil(col/block_size);
    int row_rem=row%block_size;
    int col_rem=col%block_size;
    struct proc_submatrix_info *submat_info = (struct proc_submatrix_info *) malloc(sizeof(struct proc_submatrix_info));
    if(submat_info==NULL){
        printf("Error in memory allocation for proc_submatrix_info\n");
        exit(1);
    }
    //ORA ogni processo si becca almeno NBR/pg_col per riga e NBC/pg_row per colonna però se NBR%pg_col!=0 si prende +1 a riga
    // e se NBC%pg_row!=0 si prende +1 a colonna
    
    

}

//This function is used to distribute block of rows of a matrix over the process cyclically 
void row_block_cyclic_distribution_old(float *mat, int row, int col, int block_size, int npc, MPI_Comm comm, int rank){
    //MPI_TYPE_INDEXED(count, array_of_blocklengths, array_of_displacements, oldtype, newtype) Potrebbe risolvere il problema di dividere l'array in blocchi di size diversa
    int num_blocks = row/block_size;
    int rem = row%block_size;
    int pid_norm=rank%npc;
    int *block_lenghts;
    int *block_displacements;
    float *recv_buff;
    int max_recv_cnt=((2*block_size)-1)*col;
    MPI_Datatype mat_blocks, cyclic;
    MPI_Aint extent, lb;

    recv_buff = (float *) malloc(max_recv_cnt*sizeof(float));
    if(recv_buff==NULL){
        printf("Error in memory allocation for recv_buff in row_block_cyclic_distribution\n");
        exit(1);
    }

    if(rem!=0) num_blocks++;

    block_lenghts = (int *) malloc(num_blocks*sizeof(int));
    if(block_lenghts==NULL){
        printf("Error in memory allocation for block_lenghts in row_block_cyclic_distribution\n");
        exit(1);
    }
    block_displacements = (int *) malloc(num_blocks*sizeof(int));
    if(block_displacements==NULL){
        printf("Error in memory allocation for block_displacements in row_block_cyclic_distribution\n");
        exit(1);
    }
    for(int i=0; i<num_blocks; i++){
        if((rem!=0)&&(i==num_blocks-1)) block_lenghts[i] = rem*col;
        else block_lenghts[i] = block_size*col;
        block_displacements[i] = i*block_size*col;
    }
    MPI_Barrier(comm);
    if(rank==0){
        printf("There are %d blocks\n", num_blocks);   
        printf("Block lenghts\n");
        for(int i=0; i<num_blocks; i++){
            printf("%d \n", block_lenghts[i]);
        }
        printf("Block displacements\n");
        for(int i=0; i<num_blocks; i++){
            printf("%d \n", block_displacements[i]);
        }
        printf("So blocks are:\n");
        for(int i=0; i<num_blocks; i++){
            printf("Block %d\n", i);
            for(int j=0; j<block_lenghts[i]; j++){
                printf("%19.2lf", mat[block_displacements[i]+j]);
            }
            printf("\n");
        }
    }

    
    MPI_Type_indexed(num_blocks, block_lenghts, block_displacements, MPI_FLOAT, &mat_blocks);
    MPI_Type_get_extent(MPI_FLOAT, &lb, &extent);
    //MPI_Type_create_resized(mat_blocks, 0, extent, &cyclic);
    //MPI_Type_commit(&cyclic);
    MPI_Scatter(mat, 1, mat_blocks, recv_buff, max_recv_cnt, MPI_FLOAT, 0, comm);

    MPI_Barrier(comm);
    printf("Rank %d received\n", rank);
    for(int i=0; i<max_recv_cnt; i++){
        printf("%19.2lf", recv_buff[i]);
    }
}

