default: build

build:
	mpicc matrix_vector.c -O3 

run:
	mpirun -np ${np} --oversubscribe ./a.out