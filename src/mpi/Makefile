default: build_cuda

build_cuda:
	nvcc -arch=sm_75 -lineinfo -Xcompiler=-O3 -Xptxas=-v -c ../cuda/kernel.cu -o cuda_kernel.o -I/opt/cuda/12.3/samples/Common/ 
	mpic++ -O3 -c ../mpi/mat_mul.cpp -o mat_mul.o 
	mpic++ -c ../mpi/main.cpp -o mpi_main.o 
	mpic++ mpi_main.o mat_mul.o cuda_kernel.o -L/opt/cuda/12.3/lib64 -lcudart -o mpi_cuda_main

build:
	mpic++ -O3 -c mat_mul.cpp -o mat_mul.o 
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib 

debug:
	mpic++ -c mat_mul.cpp -o mat_mul.o -DDEBUG
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib

debug_el:
	mpic++ -c mat_mul.cpp -o mat_mul.o -DDEBUG_ELEMENT
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib

audit:
	mpic++ -c mat_mul.cpp -o mat_mul.o -DAUDIT
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib 

check:
	mpic++ -c mat_mul.cpp -o mat_mul.o -DCHECK_RESULT
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib 

clean:
	rm -f *.o *.a *.out

#np is number of processes, pgr and pgc are process grid number of rows and columns pgr*pgc must be equal to np, bs is block size,
#path1 is the path of matrix A, row1 and col1 are the number of rows and columns of matrix A, 
#path2 is the path of matrix B, row2 and col2 are the number of rows and columns of matrix B, 
#path3 is the path of matrix C, path3_check is the path of matrix C to check the result, out_path is the path of output file
#blocked_mul is 1 if you want to use blocked matrix multiplication, 0 otherwise
run:
	mpirun -n ${np} ./a.out ${pgr} ${pgc} ${bs} ${path1} ${row1} ${col1} ${path2} ${row2} ${col2} ${path3} ${path3_check} ${out_path} ${blocked_mul}

runt4:
	mpirun -n 1 ./a.out 1 1 2 ../mpi/data/matrix/bin/mat1_500x500.bin 500 500 ../mpi/data/matrix/bin/mat2_500x500.bin 500 500 ../mpi/data/matrix/bin/mat3_500x500.bin ../mpi/data/matrix/bin/mat3_500x500_check.bin ../mpi/data/out/res_mpi_cuda.csv 1