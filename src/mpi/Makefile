default: build

build:
	gcc -c util/utils.c -o util/utils.o
	ar rcs libmy_lib.a util/utils.o
	mpicc mat_mul.c -O3 -L. -lmy_lib

debug:
	mpicc -DDEBUG mat_mul.c -O3 

#pgr and pgc are process grid number of rows and columns
run:
	mpirun -np ${np} --oversubscribe ./a.out ${pgr} ${pgc} ${bs} ${path1} ${row1} ${col1} ${path2} ${row2} ${col2}

runt1:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/mat1_7x5 7 5 data/matrix/mat2_5x7 5 7