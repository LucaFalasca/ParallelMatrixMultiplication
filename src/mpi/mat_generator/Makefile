default: build

build:
	gcc -c ../util/utils.c -o ../util/utils.o
	gcc -c mat_generator.c -o mat_generator.o
	ar rcs libmy_lib.a ../util/utils.o mat_generator.o 
	gcc main.c -O3 -L. -lmy_lib -o main

run:
	./main ${name1} ${row1} ${col1} ${name2} ${row2} ${col2}

runt:
	./main ../data/matrix/ ${row1} ${col1} ../data/matrix/ ${row2} ${col2}