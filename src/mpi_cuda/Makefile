build: 
	mpicc -c mpi_prova.cpp -o mpi_prova.o 
	nvcc -arch=sm_75 -c cuda/cuda_prova.cu -o cuda_prova.o -I/opt/cuda/12.3/samples/Common/
	mpicc mpi_prova.o cuda_prova.o -L/opt/cuda/12.3/lib64 -lcudart -o program
run:
	mpirun -n 2 ./program