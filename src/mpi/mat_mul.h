typedef struct block_info
{
    int num_row;        // Number of rows in the block
    int num_col;        // Number of columns in the block
    int row_start_gidx; // Global index of the first row in the block
    int col_start_gidx; // Global index of the first column in the block
} block_info;

typedef struct submat_info
{
    float *submat;                      // Pointer to the submat for this process
    int num_blocks_per_row;             // Number of blocks per row for this process
    int num_blocks_per_col;             // Number of blocks per column for this process
    int submat_row;                     // Number of rows of the submat for this process (aggregation of the blocks)
    int submat_col;                     // Number of columns of the submat for this process (aggregation of the blocks)
    struct block_info **ist_block_info; // Array of block info for this process blocks
} submat_info;

typedef struct comm_info
{
    MPI_Comm comm;  // MPI communicator
    int rank;       // Rank of the process in the communicator
    int size;       // Number of processes in the communicator
    int pg_row_idx; // Process row index in the process grid
    int pg_col_idx; // Process column index in the process grid
} comm_info;

extern void column_blocked_matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2);
extern void matrix_multiply(float *mat1, float *mat2, float *res, int r1, int c1, int c2, bool res_zero);
extern float *check_result(char mat_a_path[128], char mat_b_path[128], char mat_c_path[128], char mat_c_path_check[128], int r1, int c1, int c2);
extern void reset_matrix_c(char mat_c_path[128], char mat_c_check_path[128]);
void set_proc_grid_info(int pg_col, struct comm_info *comm_info);
void compute_block_info(int row, int col, int row_block_size, int col_block_size, int pg_row, int pg_col, struct comm_info *comm_info, struct submat_info *submat_info);
void compute_row_block_info(int row, int col, int row_block_size, int pg_row, int pg_col, struct comm_info *comm_info, struct submat_info *submat_info);
void block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info);
void row_block_cyclic_distribution(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info);
void create_row_comm(int pg_col, struct comm_info *comm_info, struct comm_info *row_comm_info);
void create_col_comm(int pg_row, struct comm_info *comm_info, struct comm_info *col_comm_info);
void create_row_leader_comm(int pg_row, int pg_col, struct comm_info *comm_info, struct comm_info *row_leader_comm_info);
void block_cyclic_write_result(char *mat_path, int row, int col, int block_size, int pg_row, int pg_col, struct submat_info *submat_info, struct comm_info *comm_info);
extern "C" void parallel_matrix_multiplication(int pg_row, int pg_col, int block_size, char *mat_a_path, int row_a, int col_a, char *mat_b_path, int row_b, int col_b, char *mat_c_path, char *mat_c_path_check, int version);
void parallel_matrix_multiplication_accelerated(int pg_row, int pg_col, int block_size, char *mat_a_path, int row_a, int col_a, char *mat_b_path, int row_b, int col_b, char *mat_c_path, char *mat_c_path_check);
void parallel_matrix_multiplication_blocked(int pg_row, int pg_col, int block_size, char *mat_a_path, int row_a, int col_a, char *mat_b_path, int row_b, int col_b, char *mat_c_path, char *mat_c_path_check);
void parallel_matrix_multiplication_naive(int pg_row, int pg_col, int block_size, char *mat_a_path, int row_a, int col_a, char *mat_b_path, int row_b, int col_b, char *mat_c_path, char *mat_c_path_check);