import subprocess

if __name__ == "__main__":
    command = "mpirun -n {} ./a.out {} {} {} data/matrix/bin/mat1_{}x{}.bin {} {} data/matrix/bin/mat2_{}x{}.bin {} {} data/matrix/bin/mat3_{}x{}.bin data/matrix/bin/mat3_{}x{}_check.bin data/matrix/bin/result_{}.csv {}"
    
    n_proc = [1, 2, 4, 6, 8, 12, 16, 20]
    pg_size = [[1, 1], [2, 1], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4], [5, 4]]
    block_size=32
    
    mat_nm_sizes = [50, 100, 500, 1000, 2048, 3000, 4096, 6000, 8192, 9000, 11000]
    mat_k_sizes =[32, 64, 128, 156]
    mat1_size = []
    mat2_size = []
    mat3_size = []
    
    
    for i in mat_nm_sizes:
        #Squared matrices
        mat1_size.append([i, i])
        mat2_size.append([i, i])
        mat3_size.append([i, i])
        
        #Rectangular matrices
        for j in mat_k_sizes:
            if i!=j:
                mat1_size.append([i, j])
                mat2_size.append([j, i])
                mat3_size.append([i, i])
    
    for i in range(0, len(n_proc)):
        for j in range(0, len(mat1_size)):
            curr_cmd=command.format(n_proc[i], pg_size[i][0], pg_size[i][1], block_size, mat1_size[j][0], mat1_size[j][1], mat1_size[j][0], mat1_size[j][1], mat2_size[j][0], mat2_size[j][1], mat2_size[j][0], mat2_size[j][1], mat3_size[j][0], mat3_size[j][1], mat3_size[j][0], mat3_size[j][1], 'blocked', 1)
            print("Launching ")
            print(curr_cmd)
            curr_cmd=command.format(n_proc[i], pg_size[i][0], pg_size[i][1], block_size, mat1_size[j][0], mat1_size[j][1], mat1_size[j][0], mat1_size[j][1], mat2_size[j][0], mat2_size[j][1], mat2_size[j][0], mat2_size[j][1], mat3_size[j][0], mat3_size[j][1], mat3_size[j][0], mat3_size[j][1], 'non_blocked', 0)
            print("Launching ")
            print(curr_cmd)
        
