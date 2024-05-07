default: build

build:
	gcc -c mat_generator.cpp -o mat_generator.o
	ar rcs libmy_lib.a mat_generator.o 
	gcc main.cpp -O3 -L. -lmy_lib -o main

debug:
	gcc -c mat_generator.c -o mat_generator.o
	ar rcs libmy_lib.a mat_generator.o 
	gcc main.c -DDEBUG -O3 -L. -lmy_lib -o main

clean:
	rm -f *.o *.a *.out

#name1 is the path of matrix A, row1 and col1 are the number of rows and columns of matrix A,
#name2 is the path of matrix B, row2 and col2 are the number of rows and columns of matrix B,
#name3 is the path of matrix C, isZero is 1 if you want to generate matrix C with only 0, 0 otherwise
#isDummy is 1 if you want to generate matrix A, B with only 1
run:
	./main ${name1} ${row1} ${col1} ${name2} ${row2} ${col2} {name3} ${isZero} ${isDummy}

#Generate matrix A, B, C with random values
runt:
	./main ../data/matrix/ ${row1} ${col1} ../data/matrix/ ${row2} ${col2} ../data/matrix/ 0 0

#Generate and C with only 0
runt0:
	./main ../data/matrix/ ${row1} ${col1} ../data/matrix/ ${row2} ${col2} ../data/matrix/ 1 0

#Generate matrix A, B with only 1 and C with only 0
runt_dummy:
	./main ../data/matrix/ ${row1} ${col1} ../data/matrix/ ${row2} ${col2} ../data/matrix/ 1 1