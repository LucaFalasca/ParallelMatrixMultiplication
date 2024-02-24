default: build

build:
	mpicc mat_mul.c -O3

debug:
	mpicc -DDEBUG mat_mul.c -O3 

#pgr and pgc are process grid number of rows and columns
run:
	mpirun -np ${np} --oversubscribe ./a.out ${pgr} ${pgc} ${bs} ${row1} ${col1} ${row2} ${col2}

run_test_rbc:
	mpirun -np 2 --oversubscribe ./a.out 1 2 2 5 3 3 5