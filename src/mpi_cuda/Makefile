default: build
build:
	nvcc -arch=sm_75 -lineinfo -Xcompiler=-O3 -Xptxas=-v -c ../cuda/kernel.cu -o cuda_prova.o -I/opt/cuda/12.3/samples/Common/ 
	mpic++ -O3 -c ../mpi/mat_mul.cpp -o mat_mul.o 
	mpic++ -c ../mpi/main.cpp -o mpi_main.o 
	mpic++ mpi_main.o mat_mul.o cuda_prova.o -L/opt/cuda/12.3/lib64 -lcu
run:
	mpirun -n ${np} ./program ${pgr} ${pgc} ${bs} ${path1} ${row1} ${col1} ${path2} ${row2} ${col2} ${path3} ${path3_check} ${out_path} ${blocked_mul}

runt:
	mpirun -n 4 ./cuda_mpi 2 2 2 ../mpi/data/matrix/bin/mat1_1024x1024.bin 1024 1024 ../mpi/data/matrix/bin/mat2_1024x1024.bin 1024 1024 ../mpi/data/matrix/bin/mat3_1024x1024.bin ../mpi/data/matrix/bin/mat3_1024x1024_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
runt1:
	mpirun -n 1 ./cuda_mpi 1 1 2 ../mpi/data/matrix/bin/mat1_1000x1000.bin 1000 1000 ../mpi/data/matrix/bin/mat2_1000x1000.bin 1000 1000 ../mpi/data/matrix/bin/mat3_1000x1000.bin ../mpi/data/matrix/bin/mat3_1000x1000_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
runt2:
	mpirun -n 1 ./cuda_mpi 1 1 2 ../mpi/data/matrix/bin/mat1_4x4.bin 4 4 ../mpi/data/matrix/bin/mat2_4x4.bin 4 4 ../mpi/data/matrix/bin/mat3_4x4.bin ../mpi/data/matrix/bin/mat3_4x4_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
runt3:
	mpirun -n 1 ./cuda_mpi 1 1 2 ../mpi/data/matrix/bin/mat1_2x2.bin 2 2 ../mpi/data/matrix/bin/mat2_2x2.bin 2 2 ../mpi/data/matrix/bin/mat3_2x2.bin ../mpi/data/matrix/bin/mat3_2x2_check.bin ../mpi/data/out/res_mpi_cuda.csv 2
