default: build

build:
	mpic++ -c mat_mul.cpp -o mat_mul.o -O3
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -O3 -L. -lmy_lib

debug:
	mpic++ -c mat_mul.cpp -o mat_mul.o -O3
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -O3 -L. -lmy_lib -DDEBUG

debug_el:
	mpic++ -c mat_mul.cpp -o mat_mul.o -O3
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -O3 -L. -lmy_lib -DDEBUG_ELEMENT

audit:
	mpic++ -c mat_mul.cpp -o mat_mul.o -O3
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -O3 -L. -lmy_lib -DAUDIT

clean:
	rm -f *.o *.a *.out

#pgr and pgc are process grid number of rows and columns
run:
	mpirun -n ${np} ./a.out ${pgr} ${pgc} ${bs} ${path1} ${row1} ${col1} ${path2} ${row2} ${col2} ${path3} ${path3_check}
