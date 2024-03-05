void compute_block_info_old(int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info);
void compute_block_info_old(int row, int col, int block_size, int pg_row, int pg_col, struct proc_info *proc_info){
    int num_blocks_row=ceil((float) col/block_size); //Number of blocks for row
    int num_blocks_col=ceil((float) row/block_size); //Number of blocks for col
    int proc_min_blocks_row, proc_min_blocks_col, proc_extra_blocks_row=0, proc_extra_blocks_col=0;
    int row_rem=row%block_size;
    int col_rem=col%block_size;
    struct proc_submatrix_info *submat_info = (struct proc_submatrix_info *) malloc(sizeof(struct proc_submatrix_info));
    if(submat_info==NULL){
        printf("Error in memory allocation for proc_submatrix_info\n");
        exit(1);
    }

    /*Ora ogni processo si becca almeno NBR/pg_col blocchi per riga e NBC/pg_row blocchi per colonna però se NBR%pg_col!=0 
    tutti i processi con indice di colonna 0 si prendono +1 blocco a riga e se NBC%pg_row!=0 tutta i processi con indice di riga 0 
    si prendono +1 blocco a colonna*/
    proc_min_blocks_row=num_blocks_row/pg_col; 
    proc_min_blocks_col=num_blocks_col/pg_row;

    //Add extra row blocks to process with row index 0
    if(((num_blocks_row%pg_col) !=0)&&(proc_info->pg_row_idx==0))
        proc_extra_blocks_row++;

    //Add extra col blocks to process with col index 0
    if(((num_blocks_col%pg_row) !=0)&&(proc_info->pg_col_idx==0)) 
        proc_extra_blocks_col++;

    submat_info->num_blocks_per_row=proc_min_blocks_row+proc_extra_blocks_row;
    submat_info->num_blocks_per_col=proc_min_blocks_col+proc_extra_blocks_col;

    /*TODO ASSEGNARE LA TAGLIA PRECISA DEI BLOCCHI, in toeria se faccio proc_extra_blocks_row -1 * block size prendo tutti quei blocchi
      che come row size hanno 2 e poi la col size dipende dal R%block_size, mentre l'ultimo se cè avra size R%block_size*C%block_size
      pero chi se lo prende il blocco fatto così? In generale solo un processo avrà quello brutto, una possibilità è una variabile
      condivisa che mantiene il max extra block corrente tra tutti i processi cosi poi ognuno la confronta e capisce se è lui che se lo becca
      un'altra possibilità è invece fare dei ragionamenti sulla size della matrice, la size dei blocchi e la size della grid
      cosi che ogni processo sappia ricostruire la size della matrice originale e capire quindi se l'ultimo blocco extra deve
      essere più piccolo o meno
    */
    submat_info->submatrix_row=(proc_min_blocks_row*block_size)+(proc_extra_blocks_row*block_size);
    submat_info->submatrix_col=(proc_min_blocks_col*block_size)+(proc_extra_blocks_col*block_size);

    
    proc_info->submat_info=submat_info;
    //printf("Process %d in grid position (%d, %d) has %d blocks\n", proc_info->rank, proc_info->pg_row_idx, proc_info->pg_col_idx, submat_info->num_blocks);
}