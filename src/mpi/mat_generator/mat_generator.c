#include<stdlib.h>
#include<stdbool.h>
#include<stdio.h>
#include<string.h>

#define MIND 200.0
#define MAXD -200.0

/* Generate a random floating point number from min to max */
float rand_from(float min, float max){
    float div = RAND_MAX / (max - min);
    return min + (rand() / div);
}

/* Generate random matrix*/
float *generate_matrix(int row, int col){
    
    float *mat=(float *) malloc(row*col*sizeof(float));
    if(mat==NULL){
        printf("Error in memory allocation for matrix generation\n");
        exit(1);
    }
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            mat[i*col+j] = rand_from(MIND, MAXD);
        }
    }
    return mat;
}

/* Write matrix to file */
void write_matrix(float *mat, int rows, int cols, char *filename){
    FILE *f;
    f = fopen(filename, "w");
    if(f==NULL){
        printf("Error in opening file for writing\n");
        exit(1);
    }
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            fprintf(f, "%f ", mat[i*cols+j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}