#include<mpi.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<stdbool.h>
#define MAXD 200
#define MIND -200
#define BLOCK_ROWS 2 
#define BLOCK_COLS 2
#define RSRC 0
#define CSRC 0


void matrix_multiply(double *mat1, double *mat2, double *res, int r1, int c1, int c2);
bool seq_check_result(double *mat1, double *mat2, double *res, int r1, int c1, int c2);
double rand_from(double min, double max);
double* generate_matrix(int r, int c);
void printMatrix(double *mat, int rows, int cols);
void row_block_cyclic_distribution(double *mat, int row, int col, int block_size, int npc, MPI_Comm comm, int rank);
//FOR 2d block cyclic distribution
//Global array coordinates (i,j) so i*c+j is the linear index of the element
//Global array element (i,j) destination process (Pr,Pc)=((RSRC+(i-1)/NR)%npr, (CSRC+(k-1)/NC)%npc)
//Global array element (i,j) local block coordinates (l,m) in process (Pr,Pc) are (l,m)=(floor((i-1)/(PR*NR)), floor((j-1)/(PC*NC))
//Global array element (i,j) local coordinates (x,y) in process (Pr,Pc) are (x,y)=((i-1)%NR, (j-1)%NC)

//FOR 1d block cyclic distribution
//Global array column j
//Global column j destination process P=((RSRC+(j-1)/NC)%p we consider blocks of NC columns
//Global column j local column block index l in process P is l=floor((j-1)/(P*NC)
//Global column j local column index x in process P is x=(j-1)%NC

int main(int argc, char *argv[]){
    int rank, size,tag=0;
    int r1, c1, r2, c2, npr, npc, block_size;
    double *mat1, *mat2, *res;
    MPI_Comm comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    /*Get parameters from cmd*/
    if(argc<8){
        printf("Usage ./a.out <nrproc> <ncproc> <blockr> <blocks> <rows1> <cols1> <rows2> <cols2> \n");
        exit(1);
    }

    //Process grid size
    npr = atoi(argv[1]);
    npc = atoi(argv[2]);

    /*Check size compatibility for process grid*/
    if((npr*npc)!=size){
        printf("Process grid size incompatible with number of processes spawned\n");
        exit(1);
    }
    //Block size
    block_size = atoi(argv[3]);

    //Matrix size
    r1 = atoi(argv[4]);
    c1 = atoi(argv[5]);
    r2 = atoi(argv[6]);
    c2 = atoi(argv[7]);

    /*Check size compatibility for matrix multiply*/
    if(c1!=r2){
        printf("Incompatible matrix size for multiplication c1!=r2\n");
        exit(1);
    }

    #ifdef DEBUG
        if(rank==0){
            printf("Number of processes: %d\n", size);
            printf("Process grid size: %d x %d\n", npr, npc);
            printf("Block size: %d x %d\n", block_size, block_size);
            printf("Matrix1 size: %d x %d\n", r1, c1);
            printf("Matrix2 size: %d x %d\n", r2, c2);
        }
    #endif

    if(rank==0){
        /*Generate matrix*/
        srand(1);
        mat1 = generate_matrix(r1, c1);
        mat2 = generate_matrix(r2, c2);
        res=(double *) malloc(r1*c2*sizeof(double));
        if(res==NULL){
            printf("Error in memory allocation for result matrix\n");
            exit(1);
        }
        memset(res, 0, r1*c2*sizeof(double)); //TODO forse non lo metto
        #ifdef DEBUG
            printf("DEBUG\n");
            printMatrix(mat1, r1, c1);
        #endif
    }
    
    row_block_cyclic_distribution(mat1, r1, c1, block_size, npc, comm, rank);

    //Scatter data
    //MPI_Scatterv(SendBuff, sendSizeArray, displacementArray, MPI_DOUBLE, recvBuff, recvSize, MPI_DOUBLE, 0, comm);

    
    /*
    // Determine the number of elements to send to each process
        //
        int send_counts[size];
        int ret_counts[size];
        int displacements[size];
        int ret_displacements[size];
        int elem_per_process = (int) (N/ (size-1))*M; //Ad ogni processo va una riga da M elementi
        int rem = N % (size-1);
        int *mat_part;
        int *local_v;
        int *ret;
        send_counts[0] = 0; //Send count will receive no data
        displacements[0] = 0; //Displacement is 0 for process 0
        ret_counts[0] = 0;
        ret_displacements[0] = 0;
        
        MPI_Barrier(comm);//Sync processes
        double t1; 
        t1 = MPI_Wtime();
        //Split data among processes except process 0
        for (int i = 1; i < size; i++) {
            send_counts[i] = elem_per_process + (i <= rem ? M : 0); //Add the remaining rows to the first processes, so M ints 
            ret_counts[i] = (elem_per_process + (i <= rem ? M : 0))/M;
            displacements[i] = (i > 0) ? displacements[i - 1] + send_counts[i - 1] : 0;
            ret_displacements[i] = (i > 0) ? ret_displacements[i - 1] + ret_counts[i - 1] : 0;
            //printf("Process %d: send count %d\n", i, send_counts[i]);
        }

        //Allocate memory for the scattered data to receive and do scatter
        mat_part = rank == 0 ? (int *)( MPI_IN_PLACE ) : (int *) malloc(send_counts[rank] * sizeof(int));
        MPI_Scatterv(mat, send_counts, displacements, MPI_INT, mat_part, send_counts[rank], MPI_INT, 0, comm);
        //Broadcast vector
        MPI_Bcast(v, M, MPI_INT, 0, comm);

    if(rank != 0){        
        ret= (int *) malloc(ret_counts[rank] * sizeof(int));

        for(int j=0; j<ret_counts[rank]; j++){
            ret[j] = 0;
            for(int i=0; i<M; i++)
                ret[j] += mat_part[j*M+i] * v[i];

        }
    }
    
    MPI_Gatherv(ret, ret_counts[rank], MPI_INT, c, ret_counts, ret_displacements, MPI_INT, 0, comm);
    if(rank==0){
        
        printf("Elapsed %lf ms\n", (MPI_Wtime()-t1)*1000);
        if(checkResult(mat, v, c))
            printf("Result is correct\n");
        else
            printf("Result is wrong\n");
    }
    */
    
    MPI_Finalize();
    return 0;
}

void matrix_multiply(double *mat1, double *mat2, double *res, int r1, int c1, int c2){
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

bool seq_check_result(double *mat1, double *mat2, double *res, int r1, int c1, int c2){
    double *correct_res;
    correct_res = (double *) malloc(r1*c2*sizeof(double));
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

/* Generate a random floating point number from min to max */
double rand_from(double min, double max) 
{
    double div = RAND_MAX / (max - min);
    return min + (rand() / div);
}

/* Generate random matrix*/
double *generate_matrix(int row, int col){
    double *mat=(double *) malloc(row*col*sizeof(double));
    if(mat==NULL){
        printf("Error in memory allocation for matrix generation\n");
        exit(1);
    }
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            mat[i*col+j] = rand_from(MIND, MAXD);
        }
    }
    return mat;
}

void printMatrix(double *mat, int row, int col) {
    printf("Matrix:\n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            //printf("%19.12lf", mat[i*col+j]);
            printf("%19.2lf", mat[i*col+j]);
        }
        printf("\n");
    }
}

//This function is used to distribute block of rows of a matrix over the process cyclically 
void row_block_cyclic_distribution(double *mat, int row, int col, int block_size, int npc, MPI_Comm comm, int rank){
    //MPI_TYPE_INDEXED(count, array_of_blocklengths, array_of_displacements, oldtype, newtype) Potrebbe risolvere il problema di dividere l'array in blocchi di size diversa
    int num_blocks = row/block_size;
    int rem = row%block_size;
    int pid_norm=rank%npc;
    int *block_lenghts;
    int *block_displacements;
    double *recv_buff;
    int max_recv_cnt=((2*block_size)-1)*col;
    MPI_Datatype mat_blocks, cyclic;
    MPI_Aint extent, lb;

    recv_buff = (double *) malloc(max_recv_cnt*sizeof(double));
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

    
    MPI_Type_indexed(num_blocks, block_lenghts, block_displacements, MPI_DOUBLE, &mat_blocks);
    MPI_Type_get_extent(MPI_DOUBLE, &lb, &extent);
    //MPI_Type_create_resized(mat_blocks, 0, extent, &cyclic);
    //MPI_Type_commit(&cyclic);
    MPI_Scatter(mat, 1, mat_blocks, recv_buff, max_recv_cnt, MPI_DOUBLE, 0, comm);

    MPI_Barrier(comm);
    printf("Rank %d received\n", rank);
    for(int i=0; i<max_recv_cnt; i++){
        printf("%19.2lf", recv_buff[i]);
    }
}

