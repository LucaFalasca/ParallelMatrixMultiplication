# ParallelMatrixMultiplication
The project is concerned with the realization of a computational core for the product between two matrices, which is then able to calculate
C ← C + AB
where A is an m × k matrix and B is a k × n matrix. For the input matrices, two main cases were considered:
1. Square matrices m = n = k;
2. Rectangular matrices m, n ≫ k; in this case k must take values
typical of the tessellation of the data k = 32, 64, 128, 156

# How to generate matrices
To generate matrices go to src/mpi/mat_generator and build with "make" then run "make runt row1=value col1=value row2=value col2=value" to generate matrices with specified values, if you want matric C to be 0 use "runt0". The generated matrices will be available in src/mpi/data/matrix/bin.
If you want you can use only the functions needed to generate the matrices etc they are defined and ready to be exported in src/mpi/mat_generator/mat_generator.cpp and src/mpi/mat_generator/mat_generator.h.

# How to build and run MPI and MPI+CUDA version
- To build go in src/mpi and run "make build_cuda"
- To run use "make run np=<num_proc> pgr=<process_grid_row_size> pgc=<process_grid_col_size> bs=<block_size> path_1=<path_of_matrix_A> row1=<rows_A> col1=<cols_A> path_2=<path_of_matrix_B> row1=<rows_B> col1=<cols_B> <path3=path_matrix_C> path3_check=<path_matrix_C_check> out_path=<where_to_write_result_csv> blocked_mul=<0, 1, 2> -> 0 is for naive product, 1 is for blocked product, 2 is for accelerated product. All the needed matrices are generated with the generator.
  
If you want you can use only the functions needed to perform the block cyclic distribution or only the product etc they are defined and ready to be exported in src/mpi/mat_mul.cpp and src/mpi/mat_mul.h.

# How to build and run CUDA version
- To build go in src/cuda and run "cmake ." and then run "make"
- To run use the cmd "matrix_mult <row_matrix_A> <col_matrix_A> <col_matrix_B>
