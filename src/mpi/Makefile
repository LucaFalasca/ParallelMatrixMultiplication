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

runt4x4:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_4x4.bin 4 4 data/matrix/bin/mat2_4x4.bin 4 4

runt_p3x3_4x4:
	mpirun -np 9 --oversubscribe ./a.out 3 3 2 data/matrix/bin/mat1_4x4.bin 4 4 data/matrix/bin/mat2_4x4.bin 4 4

runt_b3_4x4:
	mpirun -np 4 --oversubscribe ./a.out 2 2 3 data/matrix/bin/mat1_4x4.bin 4 4 data/matrix/bin/mat2_4x4.bin 4 4

runt_b4_4x4:
	mpirun -np 4 --oversubscribe ./a.out 2 2 4 data/matrix/bin/mat1_4x4.bin 4 4 data/matrix/bin/mat2_4x4.bin 4 4

runt5x5:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_5x5.bin 5 5 data/matrix/bin/mat2_5x5.bin 5 5

runt_p2x3_5x5:
	mpirun -np 6 --oversubscribe ./a.out 2 3 2 data/matrix/bin/mat1_5x5.bin 5 5 data/matrix/bin/mat2_5x5.bin 5 5

runt_p3x2_5x5:
	mpirun -np 6 --oversubscribe ./a.out 3 2 2 data/matrix/bin/mat1_5x5.bin 5 5 data/matrix/bin/mat2_5x5.bin 5 5

runt_p3x3_5x5:
	mpirun -np 9 --oversubscribe ./a.out 3 3 2 data/matrix/bin/mat1_5x5.bin 5 5 data/matrix/bin/mat2_5x5.bin 5 5

runt_b3_5x5:
	mpirun -np 4 --oversubscribe ./a.out 2 2 3 data/matrix/bin/mat1_5x5.bin 5 5 data/matrix/bin/mat2_5x5.bin 5 5

runt7x5:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_7x5.bin 7 5 data/matrix/bin/mat2_5x7.bin 5 7

runt5x7:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_5x7.bin 5 7 data/matrix/bin/mat2_7x5.bin 7 5

runt16x8:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_16x8.bin 16 8 data/matrix/bin/mat2_8x16.bin 8 16

runt_p3x3_16x8:
	mpirun -np 9 --oversubscribe ./a.out 3 3 2 data/matrix/bin/mat1_16x8.bin 16 8 data/matrix/bin/mat2_8x16bin 8 16

runt_p3x2_16x8:
	mpirun -np 6 --oversubscribe ./a.out 3 2 2 data/matrix/bin/mat1_16x8.bin 16 8 data/matrix/bin/mat2_8x16.bin 8 16

runt_b3_16x8:
	mpirun -np 4 --oversubscribe ./a.out 2 2 3 data/matrix/bin/mat1_16x8.bin 16 8 data/matrix/bin/mat2_8x16.bin 8 16

runt_p4x4_16x8:
	mpirun -np 16 --oversubscribe ./a.out 4 4 2 data/matrix/bin/mat1_7x5.bin 16 8 data/matrix/bin/mat2_5x7.bin 8 16

runt_b3_p3x3_16x8:
	mpirun -np 9 --oversubscribe ./a.out 3 3 3 data/matrix/bin/mat1_16x8.bin 16 8 data/matrix/bin/mat2_8x16.bin 8 16

runt_p3x3_7x5:
	mpirun -np 9 --oversubscribe ./a.out 3 3 2 data/matrix/bin/mat1_7x5.bin 7 5 data/matrix/bin/mat2_5x7.bin 5 7

runt_p4x3_7x5:
	mpirun -np 12 --oversubscribe ./a.out 4 3 2 data/matrix/bin/mat1_7x5.bin 7 5 data/matrix/bin/mat2_5x7.bin 5 7

runt_p3x3_7x7:
	mpirun -np 9 --oversubscribe ./a.out 3 3 2 data/matrix/bin/mat1_7x7.bin 7 7 data/matrix/bin/mat2_7x7.bin 7 7

runt_b3_7x7:
	mpirun -np 4 --oversubscribe ./a.out 2 2 3 data/matrix/bin/mat1_7x7.bin 7 7 data/matrix/bin/mat2_7x7.bin 7 7

runt_b3_p3x3_7x7:
	mpirun -np 9 --oversubscribe ./a.out 3 3 3 data/matrix/bin/mat1_7x7.bin 7 7 data/matrix/bin/mat2_7x7.bin 7 7

runt_7x7:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_7x7.bin 7 7 data/matrix/bin/mat2_7x7.bin 7 7

runt_p3x3_8x8:
	mpirun -np 9 --oversubscribe ./a.out 3 3 2 data/matrix/bin/mat1_8x8.bin 8 8 data/matrix/bin/mat2_7x7.bin 8 8

runt_b3_p3x3_8x8:
	mpirun -np 9 --oversubscribe ./a.out 3 3 3 data/matrix/bin/mat1_8x8.bin 8 8 data/matrix/bin/mat2_7x7.bin 8 8

runt_b3_8x8:
	mpirun -np 4 --oversubscribe ./a.out 2 2 3 data/matrix/bin/mat1_8x8.bin 8 8 data/matrix/bin/mat2_7x7.bin 8 8

runt_p3x3_8x6:
	mpirun -np 9 --oversubscribe ./a.out 3 3 2 data/matrix/bin/mat1_8x6.bin 8 6 data/matrix/bin/mat2_6x8.bin 6 8

runt_b3_p3x3_8x6:
	mpirun -np 9 --oversubscribe ./a.out 3 3 3 data/matrix/bin/mat1_8x6.bin 8 6 data/matrix/bin/mat2_6x8.bin 6 8

runt_b3_8x6:
	mpirun -np 4 --oversubscribe ./a.out 2 2 3 data/matrix/bin/mat1_8x6.bin 8 6 data/matrix/bin/mat2_6x8.bin 6 8

runt_8x7:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_8x7.bin 8 7 data/matrix/bin/mat2_7x8.bin 7 8

runt_p3x3_8x7:
	mpirun -np 9 --oversubscribe ./a.out 3 3 2 data/matrix/bin/mat1_8x6.bin 8 7 data/matrix/bin/mat2_6x8.bin 7 8

runt_b3_p3x3_8x7:
	mpirun -np 9 --oversubscribe ./a.out 3 3 3 data/matrix/bin/mat1_8x6.bin 8 7 data/matrix/bin/mat2_6x8.bin 7 8

runt_b3_8x7:
	mpirun -np 4 --oversubscribe ./a.out 2 2 3 data/matrix/bin/mat1_8x6.bin 8 7 data/matrix/bin/mat2_6x8.bin 7 8

runt_5x7_7x4:
	mpirun -np 4 --oversubscribe ./a.out 2 2 2 data/matrix/bin/mat1_5x7.bin 5 7 data/matrix/bin/mat2_7x4.bin 7 4 data/matrix/bin/mat3_5x4.bin data/matrix/bin/mat3_5x4_check.bin

