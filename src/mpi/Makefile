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


runt5x5:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_7x5.bin 5 5 data/matrix/bin/mat2_5x7.bin 5 5

runt7x5:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_7x5.bin 7 5 data/matrix/bin/mat2_5x7.bin 5 7

runt16x8:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_7x5.bin 16 8 data/matrix/bin/mat2_5x7.bin 8 16