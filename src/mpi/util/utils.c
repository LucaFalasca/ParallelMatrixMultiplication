#include<stdio.h>
#include"utils.h"

void printMatrix(float *mat, int row, int col) {
    printf("Matrix:\n");
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", mat[i*col+j]);
        }
        printf("\n");
    }
}