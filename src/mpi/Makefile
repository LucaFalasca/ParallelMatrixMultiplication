default: build

build:
	mpic++ -O3 -c mat_mul.cpp -o mat_mul.o 
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib 

debug:
	mpic++ -c mat_mul.cpp -o mat_mul.o -DDEBUG
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib

debug_el:
	mpic++ -c mat_mul.cpp -o mat_mul.o -DDEBUG_ELEMENT
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib

audit:
	mpic++ -c mat_mul.cpp -o mat_mul.o -DAUDIT
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib 

check:
	mpic++ -c mat_mul.cpp -o mat_mul.o -DCHECK_RESULT
	ar rcs libmy_lib.a mat_mul.o
	mpic++ main.cpp -L. -lmy_lib 

clean:
	rm -f *.o *.a *.out

#pgr and pgc are process grid number of rows and columns
run:
	mpirun -n ${np} ./a.out ${pgr} ${pgc} ${bs} ${path1} ${row1} ${col1} ${path2} ${row2} ${col2} ${path3} ${path3_check}
