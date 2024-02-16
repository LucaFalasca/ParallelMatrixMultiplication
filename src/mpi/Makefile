default: build

build:
	mpicc matrix_vector.c -O3 

#pgr and pgc are process grid number of rows and columns
run:
	mpirun -np ${np} --oversubscribe ./a.out ${pgr} ${pgc}
