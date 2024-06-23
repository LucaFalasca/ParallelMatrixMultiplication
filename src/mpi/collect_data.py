import subprocess

def generate_sizes(mat_nm_sizes, mat_k_sizes):
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
    return mat1_size, mat2_size, mat3_size

def run_experiment(n_proc, pg_size, block_size, mat1_size, mat2_size, mat3_size, blocked_string, blocked_flag):
    command = "mpirun -n {} ./a.out {} {} {} data/matrix/bin/mat1_{}x{}.bin {} {} data/matrix/bin/mat2_{}x{}.bin {} {} data/matrix/bin/mat3_{}x{}.bin data/matrix/bin/mat3_{}x{}_check.bin data/out/result_{}.csv {}"
    for i in range(0, len(n_proc)):
        for j in range(0, len(mat1_size)):
            if not blocked_flag:
                if (n_proc[i]==1 or n_proc[i]==2) and (mat1_size[j][0]==8192 and mat1_size[j][1]==8192):
                    continue
            curr_cmd=command.format(n_proc[i], pg_size[i][0], pg_size[i][1], block_size, mat1_size[j][0], mat1_size[j][1], mat1_size[j][0], mat1_size[j][1], mat2_size[j][0], mat2_size[j][1], mat2_size[j][0], mat2_size[j][1], mat3_size[j][0], mat3_size[j][1], mat3_size[j][0], mat3_size[j][1], blocked_string, blocked_flag)
            print("Launching ")
            print(curr_cmd)
            # Launch the C program using subprocess
            process = subprocess.Popen(curr_cmd, shell=True)
            # Wait for the process to finish
            process.wait()

if __name__ == "__main__":
    n_proc = [1, 2, 4, 6, 8, 12, 16, 20]
    pg_size = [[1, 1], [2, 1], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4], [5, 4]]
    block_size=32
    
    #mat_nm_sizes = [32, 64 ,128, 256, 400, 512, 800, 1024, 2048, 3000, 4096, 6000, 8192, 10000]
    mat_nm_sizes = [32, 64 ,100, 250, 400, 600, 800, 1000, 2000, 3000, 4000, 6000, 8000, 10000]
    mat_k_sizes =[32, 64, 128, 156]
    mat_k_sizes =[32, 64, 128, 156]
    
    mat1_size, mat2_size, mat3_size = generate_sizes(mat_nm_sizes, mat_k_sizes)
    
    #Unblocked matrix multiplication
    run_experiment(n_proc, pg_size, block_size, mat1_size, mat2_size, mat3_size, 'non_blocked', 0)
    
    #Blocked matrix multiplication
    run_experiment(n_proc, pg_size, block_size, mat1_size, mat2_size, mat3_size, 'blocked', 1)

    run_experiment(n_proc, pg_size, block_size, mat1_size, mat2_size, mat3_size, 'accelerated', 2)
    
        
