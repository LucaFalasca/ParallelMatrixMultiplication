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

struct block_info{
    int num_row; //Number of rows in the block
    int num_col; //Number of columns in the block
    int row_start_gidx; //Global index of the first row in the block
    int col_start_gidx; //Global index of the first column in the block
} block_info;

struct proc_submatrix_info{
    float *submatrix; //Pointer to the submatrix for this process
    int num_blocks_per_row; //Number of blocks per row for this process
    int num_blocks_per_col; //Number of blocks per column for this process
    int submatrix_row; //Number of rows of the submatrix for this process (aggregation of the blocks)
    int submatrix_col; //Number of columns of the submatrix for this process (aggregation of the blocks)
    struct block_info **ist_block_info; //Array of block info for this process blocks
} proc_submatrix_info;

struct proc_info{
    int rank; //Process rank
    int size; //Number of processes
    MPI_Comm comm; //MPI communicator
    int pg_row_idx; //Process row index in the process grid 
    int pg_col_idx; //Process column index in the process grid
    struct proc_submatrix_info *submat_info; //Pointer to the submatrix info
} proc_info;

void matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2);
bool seq_check_result(float *mat1, float *mat2, float *res, int r1, int c1, int c2);
void row_block_cyclic_distribution_old(float *mat, int row, int col, int block_size, int npc, MPI_Comm comm, int rank);
void compute_block_info_old(int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info);
void compute_block_info(int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info);
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
        if(proc_info->rank==0){
            printf("DEBUG -> Number of processes: %d\n", proc_info->size);
            printf("DEBUG -> Process grid size: %d x %d\n", pg_row, pg_col);
            printf("DEBUG -> Block size: %d x %d\n", block_size, block_size);
            printf("DEBUG -> Matrix A path %s size: %d x %d\n", mat_a_path, row_a, col_a);
            printf("DEBUG -> Matrix B path %s size: %d x %d\n", mat_b_path, row_b, col_b);
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
    compute_block_info(row, col, block_size, pg_row, pg_col, proc_info);

    printf("Rank %d in grid position (%d, %d) has %d x %d submatrix\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, proc_info->submat_info->submatrix_row, proc_info->submat_info->submatrix_col);
    
    recv_block = (float *) malloc(proc_info->submat_info->submatrix_row*proc_info->submat_info->submatrix_col*sizeof(float));
    recv_block=memset(recv_block, 0, proc_info->submat_info->submatrix_row*proc_info->submat_info->submatrix_col*sizeof(float));

    /*
    Creazione del tipo di dato per la matrice distribuita, ogni processo vedrà solo la sua porzione di matrice,
    la porzione viene definita tramite block cyclic distribution
    */
    MPI_Type_create_darray(proc_info->size, proc_info->rank, 2, dims, distribs, dargs, proc_dims, MPI_ORDER_C, MPI_FLOAT, &mat_darray);
    MPI_Type_commit(&mat_darray);

    //Apertura collettiva del file
    #ifdef DEBUG
        if(proc_info->rank==0){
            printf("DEBUG -> Opening file %s\n", mat_path);
        }
    #endif
    MPI_File_open(proc_info->comm, mat_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_file);
    if (mat_file == MPI_FILE_NULL) {
	    printf("Error opening file %s in block cyclic distribution\n", mat_path);
        exit(1);
    }

    //Ogni processo ha una visione della matrice specificata dal darray creato in precedenza
    MPI_File_set_view(mat_file, 2*sizeof(float), MPI_FLOAT, mat_darray, "native", MPI_INFO_NULL);

    MPI_File_read_all(mat_file, recv_block, proc_info->submat_info->submatrix_row* proc_info->submat_info->submatrix_col, MPI_FLOAT, &status);

    MPI_File_close(&mat_file);

    #ifdef DEBUg
    for(int i=0; i<(proc_info->submat_info->submatrix_row)*(proc_info->submat_info->submatrix_col); i++){
        printf("DEBUG -> Rank (%d, %d): %f\n", proc_info->pg_row_idx, proc_info->pg_col_idx, recv_block[i]);
    }
    #endif

    //MPI_Type_free(&mat_darray);
    
}

//Compute the process coordinates in the processg grid
void set_proc_grid_info(struct proc_info* proc_info, int pg_col){
    proc_info->pg_row_idx = proc_info->rank/pg_col;
    proc_info->pg_col_idx = proc_info->rank%pg_col;
}

void compute_block_info(int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info){
    int submatrix_elem_per_row=0, submatrix_elem_per_col=0;
    int num_block_per_row_per_proc=0, num_block_per_col_per_proc=0;
    int num_extra_block_per_row=0, num_extra_block_per_col=0;
    int temp, rem_block_per_col, rem_block_per_row;

    struct proc_submatrix_info *submat_info = (struct proc_submatrix_info *) malloc(sizeof(struct proc_submatrix_info));
    if(submat_info==NULL){
        printf("Error in memory allocation for proc_submatrix_info in compute_block_info\n");
        exit(1);
    }

    /*I blocchi base sono intesi in numero di griglie complete e.g una matrice 16x7 divisa in blocchi 2x2 e process grid 2x2
     avrà solo un blocco completo per processo presente in ogni riga, quindi P00 avrà il blocco base 0 ed il blocco completo ma non base 2, 
     mentre P01 avrà il blocco base 1 e il blocco incompleto 3 che avrà una sola colonna
    */
    num_block_per_row_per_proc=col/(block_size*pg_col); //Per ora sono solo blocchi base
    num_block_per_col_per_proc=row/(block_size*pg_row);

    //Calcolo dei blocchi extra per riga
    rem_block_per_row=col%block_size;
    temp=((int)ceil((float)col/block_size))%pg_col;
    
    if((rem_block_per_row!=0)&&(temp!=0))
        num_extra_block_per_row=temp;
    else
        num_extra_block_per_row=pg_col;

    //Calcolo dei blocchi extra per colonna
    rem_block_per_col=row%block_size;
    temp=((int)ceil((float)row/block_size))%pg_row;

    if(rem_block_per_col!=0&&(temp!=0))
        num_extra_block_per_col=temp;
    else
        num_extra_block_per_col=pg_row;
    
    //num_extra_block_per_row=((int)ceil((float)col/block_size))%pg_col;
    //num_extra_block_per_col=((int)ceil((float)row/block_size))%pg_row;
    //printf("%d\n", (int)ceil((float)row/block_size)%pg_row); // TODO BUG 7/2%2=4%2 fa 0 mhh

    #ifdef DEBUG
        if(proc_info->rank==0){
            printf("DEBUG -> Number of base blocks per row per process: %d\n", num_block_per_row_per_proc);
            printf("DEBUG -> Number of base blocks per col per process: %d\n", num_block_per_col_per_proc);
            printf("DEBUG -> Number of extra blocks per row: %d\n", num_extra_block_per_row);
            printf("DEBUG -> Number of extra blocks per col: %d\n", num_extra_block_per_col);
        }
    #endif

    //Assign extra block to first processes cyclically
    if(proc_info->pg_col_idx<num_extra_block_per_row){
        num_block_per_row_per_proc++;
    }    
    if(proc_info->pg_row_idx<num_extra_block_per_col){
        num_block_per_col_per_proc++;
    }
    

    submatrix_elem_per_row=num_block_per_row_per_proc*block_size;
    submatrix_elem_per_col=num_block_per_col_per_proc*block_size;

    if(proc_info->pg_col_idx==num_extra_block_per_row-1)
        submatrix_elem_per_row-=block_size-(rem_block_per_row);
    
    if(proc_info->pg_row_idx==num_extra_block_per_col-1)
        submatrix_elem_per_col-=block_size-(rem_block_per_col); 

    #ifdef DEBUG
        if(proc_info->rank==1){
            printf("DEBUG -> Number of blocks per row: %d\n", num_block_per_row_per_proc);
            printf("DEBUG -> Number of blocks per col: %d\n", num_block_per_col_per_proc);
            printf("DEBUG -> Submatrix row size: %d\n", submatrix_elem_per_row);
            printf("DEBUG -> Submatrix col size: %d\n", submatrix_elem_per_col);
        }
    #endif

    submat_info->num_blocks_per_row=num_block_per_row_per_proc;  
    submat_info->num_blocks_per_col=num_block_per_col_per_proc;
    submat_info->submatrix_row=submatrix_elem_per_col;
    submat_info->submatrix_col=submatrix_elem_per_row;
    proc_info->submat_info=submat_info;
}

void compute_block_info_old(int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info){
    int num_blocks_row=ceil((float) col/block_size); //Number of blocks for row
    int num_blocks_col=ceil((float) row/block_size); //Number of blocks for col
    int proc_min_blocks_row, proc_min_blocks_col, proc_extra_blocks_row=0, proc_extra_blocks_col=0;
    int row_rem=row%block_size;
    int col_rem=col%block_size;
    struct proc_submatrix_info *submat_info = (struct proc_submatrix_info *) malloc(sizeof(struct proc_submatrix_info));
    if(submat_info==NULL){
        printf("Error in memory allocation for proc_submatrix_info\n");
        exit(1);
    }

    /*Ora ogni processo si becca almeno NBR/pg_col blocchi per riga e NBC/pg_row blocchi per colonna però se NBR%pg_col!=0 
    tutti i processi con indice di colonna 0 si prendono +1 blocco a riga e se NBC%pg_row!=0 tutta i processi con indice di riga 0 
    si prendono +1 blocco a colonna*/
    proc_min_blocks_row=num_blocks_row/pg_col; 
    proc_min_blocks_col=num_blocks_col/pg_row;

    //Add extra row blocks to process with row index 0
    if(((num_blocks_row%pg_col) !=0)&&(proc_info->pg_row_idx==0))
        proc_extra_blocks_row++;

    //Add extra col blocks to process with col index 0
    if(((num_blocks_col%pg_row) !=0)&&(proc_info->pg_col_idx==0)) 
        proc_extra_blocks_col++;

    submat_info->num_blocks_per_row=proc_min_blocks_row+proc_extra_blocks_row;
    submat_info->num_blocks_per_col=proc_min_blocks_col+proc_extra_blocks_col;

    /*TODO ASSEGNARE LA TAGLIA PRECISA DEI BLOCCHI, in toeria se faccio proc_extra_blocks_row -1 * block size prendo tutti quei blocchi
      che come row size hanno 2 e poi la col size dipende dal R%block_size, mentre l'ultimo se cè avra size R%block_size*C%block_size
      pero chi se lo prende il blocco fatto così? In generale solo un processo avrà quello brutto, una possibilità è una variabile
      condivisa che mantiene il max extra block corrente tra tutti i processi cosi poi ognuno la confronta e capisce se è lui che se lo becca
      un'altra possibilità è invece fare dei ragionamenti sulla size della matrice, la size dei blocchi e la size della grid
      cosi che ogni processo sappia ricostruire la size della matrice originale e capire quindi se l'ultimo blocco extra deve
      essere più piccolo o meno
    */
    submat_info->submatrix_row=(proc_min_blocks_row*block_size)+(proc_extra_blocks_row*block_size);
    submat_info->submatrix_col=(proc_min_blocks_col*block_size)+(proc_extra_blocks_col*block_size);

    
    proc_info->submat_info=submat_info;
    //printf("Process %d in grid position (%d, %d) has %d blocks\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, submat_info->num_blocks);
}

