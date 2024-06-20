build: 
	mpic++ -O3 -c ../mpi/mat_mul.cpp -o mat_mul.o 
	ar rcs libmy_lib.a mat_mul.o
	mpic++ -c ../mpi/main.cpp -L../mpi -lmy_lib -o mpi_prova.o 
	nvcc -arch=sm_75 -lineinfo -Xcompiler=-O3 -Xptxas=-v -c cuda_prova.cu -o cuda_prova.o -I/opt/cuda/12.3/samples/Common/ 
	mpic++ mpi_prova.o cuda_prova.o -L/opt/cuda/12.3/lib64 -L. -lmy_lib -lcudart -o program

prova:
	mpic++ -c mpi_prova.cpp -o mpi_prova.o 
	nvcc -arch=sm_75 -lineinfo -Xcompiler=-O3 -Xptxas=-v -c ../cuda/matrix_mult.cu -o cuda_prova.o -I/opt/cuda/12.3/samples/Common/ 
	mpic++ mpi_prova.o cuda_prova.o -L/opt/cuda/12.3/lib64 -lcudart -o program2
run:
	mpirun -n ${np} ./program ${pgr} ${pgc} ${bs} ${path1} ${row1} ${col1} ${path2} ${row2} ${col2} ${path3} ${path3_check} ${out_path} ${blocked_mul}

run2:
	mpirun -n 2 ./program2