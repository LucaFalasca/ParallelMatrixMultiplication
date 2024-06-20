#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include"mat_generator.h"



void printMatrix(float *mat, int row, int col);

int main(int argc, char *argv[]){
    int r1, c1, r2, c2, isZero, isDummy;
    char mat1path_txt[128], mat2path_txt[128], mat3_path_txt[128],mat1path_bin[128], mat2path_bin[128], mat2_trasp_path_bin[128], mat3_path_bin[128], mat3_path_bin_check[128];
    float *mat1, *mat2, *mat2_trasp, *mat3;
    if(argc<10){
        printf("Usage ./a.out <mat1path> <rows1> <cols1> <mat2path> <rows2> <cols2> <mat3path> <isZero> <isDummy>\n");
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

    //Matrix 3
    strcpy(mat3_path_txt, argv[7]);
    strcpy(mat3_path_bin, argv[7]);
    strcpy(mat3_path_bin_check, argv[7]);
    isZero = atoi(argv[8]);
    isDummy = atoi(argv[9]);

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

    strcat(mat3_path_txt, "txt/mat3_");
    strcat(mat3_path_txt, argv[2]);
    strcat(mat3_path_txt, "x");
    strcat(mat3_path_txt, argv[6]);
    strcat(mat3_path_txt, ".txt");

    strcat(mat3_path_bin, "bin/mat3_");
    strcat(mat3_path_bin, argv[2]);
    strcat(mat3_path_bin, "x");
    strcat(mat3_path_bin, argv[6]);
    strcat(mat3_path_bin, ".bin");

    strcat(mat3_path_bin_check, "bin/mat3_");
    strcat(mat3_path_bin_check, argv[2]);
    strcat(mat3_path_bin_check, "x");
    strcat(mat3_path_bin_check, argv[6]);
    strcat(mat3_path_bin_check, "_check");
    strcat(mat3_path_bin_check, ".bin");

    printf("Generating matrices of size %d x %d, %d x %d and %d x %d\n", r1, c1, r2, c2, r1, c2);

    srand(1);
    if(isDummy){
        mat1 = generate_dummy_matrix(r1, c1, 1.0);
        mat2 = generate_dummy_matrix(r2, c2, 1.0);
    }
    else{
        mat1 = generate_matrix(r1, c1);
        mat2 = generate_matrix(r2, c2);
    }

    if((!isZero)&&(!isDummy))
        mat3 = generate_matrix(r1, c2);
    else if(isDummy)
        generate_dummy_matrix(r1, c2, 1.0);
    else
        generate_dummy_matrix(r1, c2, 0.0);
    

    #ifdef DEBUG
        printMatrix(mat1, r1, c1);
        printMatrix(mat2, r2, c2);
        printMatrix(mat3, r1, c2);
    #endif

    printf("Writing matrix 1 to %s and %s\n", mat1path_txt, mat1path_bin);
    write_matrix(mat1, r1, c1, mat1path_txt, mat1path_bin);

    printf("Writing matrix 2 to %s and %s\n", mat2path_txt, mat2path_bin);
    write_matrix(mat2, r2, c2, mat2path_txt, mat2path_bin);

    printf("Writing matrix 3 to %s and %s\n", mat3_path_txt, mat3_path_bin);
    write_matrix(mat3, r1, c2, mat3_path_txt, mat3_path_bin);

    printf("Writing matrix 3 to %s\n", mat3_path_bin_check);
    write_matrix(mat3, r1, c2, NULL, mat3_path_bin_check);
    

}

void printMatrix(float *mat, int row, int col) {
    printf("Matrix:\n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", mat[i*col+j]);
        }
        printf("\n");
    }
}