#include<stdlib.h>
#include<stdbool.h>
#include<stdio.h>
#include<string.h>

#define MIND 0.0
#define MAXD 100.0

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

float *generate_dummy_matrix(int row, int col, float value){
    
    float *mat=(float *) malloc(row*col*sizeof(float));
    if(mat==NULL){
        printf("Error in memory allocation for matrix generation\n");
        exit(1);
    }
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            mat[i*col+j] = value;
        }
    }
    return mat;
}

/* Write matrix to file */
void write_matrix(float *mat, int rows, int cols, char *matpath_txt, char *matpath_bin){
    FILE *f_txt;
    FILE *f_binary;
    /*if(matpath_txt!=NULL)
        f_txt = fopen(matpath_txt, "w");*/ 
    f_binary = fopen(matpath_bin, "w+");

    /*if(f_txt==NULL){
        printf("Error in opening file %s for writing\n", matpath_txt);
        exit(1);
    }*/
    if(f_binary==NULL){
        printf("Error in opening file %s for writing\n", matpath_bin);
        exit(1);
    }

    //Write txt file
    /*if(matpath_txt!=NULL){
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                fprintf(f_txt, "%f ", mat[i*cols+j]);
            }
            fprintf(f_txt, "\n");
        }
    }*/

    // Write rows number as first element of the binary file
    if(fwrite(&rows, sizeof(int), 1, f_binary)==0){
        exit(0);
    }
    
    // Write cols number as second element of the binary file
    if(fwrite(&cols, sizeof(int), 1, f_binary)==0){
        exit(0);
    }

    //Write binary file
    if(fwrite(mat, sizeof(float), rows*cols, f_binary)==0){
        printf("Error in writing to binary file %s\n", matpath_bin);
        exit(1);
    }

    /*if(matpath_txt!=NULL)
        fclose(f_txt);*/
    fclose(f_binary);
}