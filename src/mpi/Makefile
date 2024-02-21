default: build

build:
	mpicc mat_mul.c -O3

debug:
	mpicc -DDEBUG mat_mul.c -O3 

#pgr and pgc are process grid number of rows and columns
run:
	mpirun -np ${np} --oversubscribe ./a.out ${row1} ${col1} ${row2} ${col2} ${pgr} ${pgc}

run_test:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 2 4 4 4 5 