default: build
build:
	nvcc -arch=sm_75 -lineinfo -Xcompiler=-O3 -Xptxas=-v -c ../cuda/kernel.cu -o cuda_kernel.o -I/opt/cuda/12.3/samples/Common/ 
	mpic++ -O3 -c ../mpi/mat_mul.cpp -o mat_mul.o 
	mpic++ -c ../mpi/main.cpp -o mpi_main.o 
	mpic++ mpi_main.o mat_mul.o cuda_kernel.o -L/opt/cuda/12.3/lib64 -lcudart

run:
	mpirun -n ${np} ./cuda_mpi_main ${pgr} ${pgc} ${bs} ${path1} ${row1} ${col1} ${path2} ${row2} ${col2} ${path3} ${path3_check} ${out_path} ${mul_version}
#np is number of processes, pgr and pgc are process grid number of rows and columns pgr*pgc must be equal to np, bs is block size,
#path1 is the path of matrix A, row1 and col1 are the number of rows and columns of matrix A, 
#path2 is the path of matrix B, row2 and col2 are the number of rows and columns of matrix B, 
#path3 is the path of matrix C, path3_check is the path of matrix C to check the result, out_path is the path of output file
#mul_vers is 2 if you want gpu accelerated multiplication, is 1 if you want to use blocked matrix multiplication, 0 otherwise

clean:
	rm *.o mpi_cuda_main

runt:
	mpirun -n 4 ./mpi_cuda_main 2 2 2 ../mpi/data/matrix/bin/mat1_1024x1024.bin 1024 1024 ../mpi/data/matrix/bin/mat2_1024x1024.bin 1024 1024 ../mpi/data/matrix/bin/mat3_1024x1024.bin ../mpi/data/matrix/bin/mat3_1024x1024_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
runt1:
	mpirun -n 1 ./mpi_cuda_main 1 1 2 ../mpi/data/matrix/bin/mat1_1000x1000.bin 1000 1000 ../mpi/data/matrix/bin/mat2_1000x1000.bin 1000 1000 ../mpi/data/matrix/bin/mat3_1000x1000.bin ../mpi/data/matrix/bin/mat3_1000x1000_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
runt2:
	mpirun -n 1 ./mpi_cuda_main 1 1 2 ../mpi/data/matrix/bin/mat1_4x4.bin 4 4 ../mpi/data/matrix/bin/mat2_4x4.bin 4 4 ../mpi/data/matrix/bin/mat3_4x4.bin ../mpi/data/matrix/bin/mat3_4x4_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
runt3:
	mpirun -n 1 ./mpi_cuda_main 1 1 2 ../mpi/data/matrix/bin/mat1_2x2.bin 2 2 ../mpi/data/matrix/bin/mat2_2x2.bin 2 2 ../mpi/data/matrix/bin/mat3_2x2.bin ../mpi/data/matrix/bin/mat3_2x2_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
runt4:
	mpirun -n 1 ./mpi_cuda_main 1 1 2 ../mpi/data/matrix/bin/mat1_16x16.bin 16 16 ../mpi/data/matrix/bin/mat2_16x16.bin 16 16 ../mpi/data/matrix/bin/mat3_16x16.bin ../mpi/data/matrix/bin/mat3_16x16_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
runt5:
	mpirun -n 1 ./mpi_cuda_main 1 1 2 ../mpi/data/matrix/bin/mat1_200x200.bin 200 200 ../mpi/data/matrix/bin/mat2_200x200.bin 200 200 ../mpi/data/matrix/bin/mat3_200x200.bin ../mpi/data/matrix/bin/mat3_200x200_check.bin ../mpi/data/out/res_mpi_cuda.csv 1
