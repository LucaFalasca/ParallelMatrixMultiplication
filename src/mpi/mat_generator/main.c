#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include"mat_generator.h"
#include"../util/utils.h"


int main(int argc, char *argv[]){
    int r1, c1, r2, c2;
    char mat1name[128], mat2name[128];
    float *mat1, *mat2;
    if(argc<7){
        printf("Usage ./a.out <mat1path> <rows1> <cols1> <mat2path> <rows2> <cols2>\n");
        exit(1);
    }
    strcpy(mat1name, argv[1]);
    r1 = atoi(argv[2]);
    c1 = atoi(argv[3]);
    strcpy(mat2name, argv[4]);
    r2 = atoi(argv[5]);
    c2 = atoi(argv[6]);
    //Append matrix size to name
    strcat(mat1name, "_");
    strcat(mat1name, argv[2]);
    strcat(mat1name, "x");
    strcat(mat1name, argv[3]);
    strcat(mat2name, "_");
    strcat(mat2name, argv[5]);
    strcat(mat2name, "x");
    strcat(mat2name, argv[6]);
    printf("Generating matrices of size %d x %d and %d x %d\n", r1, c1, r2, c2);

    srand(1);
    mat1 = generate_matrix(r1, c1);
    mat2 = generate_matrix(r2, c2);
    printMatrix(mat1, r1, c1);
    printMatrix(mat2, r2, c2);
    printf("Writing matrices to %s and %s\n", mat1name, mat2name);
    write_matrix(mat1, r1, c1, mat1name);
    write_matrix(mat2, r2, c2, mat2name);

}