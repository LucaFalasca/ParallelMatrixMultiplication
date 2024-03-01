#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include"mat_generator.h"
#include"../util/utils.h"


int main(int argc, char *argv[]){
    int r1, c1, r2, c2;
    char mat1path_txt[128], mat2path_txt[128], mat1path_bin[128], mat2path_bin[128];
    float *mat1, *mat2;
    if(argc<7){
        printf("Usage ./a.out <mat1path> <rows1> <cols1> <mat2path> <rows2> <cols2>\n");
        exit(1);
    }

    //Matrix 1
    strcpy(mat1path_txt, argv[1]);
    strcpy(mat1path_bin, argv[1]);
    r1 = atoi(argv[2]);
    c1 = atoi(argv[3]);

    //Matrix 2
    strcpy(mat2path_txt, argv[4]);
    strcpy(mat2path_bin, argv[4]);
    r2 = atoi(argv[5]);
    c2 = atoi(argv[6]);

    //Append matrix size to name
    strcat(mat1path_txt, "txt/mat1_");
    strcat(mat1path_txt, argv[2]);
    strcat(mat1path_txt, "x");
    strcat(mat1path_txt, argv[3]);
    strcat(mat1path_txt, ".txt");

    strcat(mat1path_bin, "bin/mat1_");
    strcat(mat1path_bin, argv[2]);
    strcat(mat1path_bin, "x");
    strcat(mat1path_bin, argv[3]);
    strcat(mat1path_bin, ".bin");

    strcat(mat2path_txt, "txt/mat2_");
    strcat(mat2path_txt, argv[5]);
    strcat(mat2path_txt, "x");
    strcat(mat2path_txt, argv[6]);
    strcat(mat2path_txt, ".txt");

    strcat(mat2path_bin, "bin/mat2_");
    strcat(mat2path_bin, argv[5]);
    strcat(mat2path_bin, "x");
    strcat(mat2path_bin, argv[6]);
    strcat(mat2path_bin, ".bin");

    printf("Generating matrices of size %d x %d and %d x %d\n", r1, c1, r2, c2);

    srand(1);
    mat1 = generate_matrix(r1, c1);
    mat2 = generate_matrix(r2, c2);

    #ifdef DEBUG
        printMatrix(mat1, r1, c1);
        printMatrix(mat2, r2, c2);
    #endif

    printf("Writing matrix 1 to %s and %s\n", mat1path_txt, mat1path_bin);
    write_matrix(mat1, r1, c1, mat1path_txt, mat1path_bin);

    printf("Writing matrix 2 to %s and %s\n", mat2path_txt, mat2path_bin);
    write_matrix(mat2, r2, c2, mat2path_txt, mat2path_bin);
}