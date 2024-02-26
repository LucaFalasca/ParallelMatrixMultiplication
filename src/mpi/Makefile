default: build

build:
	gcc -c util/utils.c -o util/utils.o
	gcc -c mat_generator/mat_generator.c -o mat_generator/mat_generator.o
	ar rcs my_lib.a util/utils.o mat_generator.o
	mpicc mat_mul.c -O3 -L. -lmy_lib -o a.out

debug:
	mpicc -DDEBUG mat_mul.c -O3 

#pgr and pgc are process grid number of rows and columns
run:
	mpirun -np ${np} --oversubscribe ./a.out ${pgr} ${pgc} ${bs} ${row1} ${col1} ${row2} ${col2}

run_test_rbc:
	mpirun -np 2 --oversubscribe ./a.out 1 2 2 5 3 3 5