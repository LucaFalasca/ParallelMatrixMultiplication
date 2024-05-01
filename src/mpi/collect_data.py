import subprocess

if __name__ == "__main__":
    command = "mpirun -n {} ./a.out {} {} {} data/matrix/bin/mat1_{}x{}.bin {} {} data/matrix/bin/mat2_{}x{}.bin {} {} data/matrix/bin/mat3_{}x{}.bin data/matrix/bin/mat3_{}x{}_check.bin"
    mat_sizes=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
    mat1_size = []
    mat2_size = []
    mat3_size = []
    n_proc = [1, 2, 4, 6, 8, 12, 16, 20]
    pg_size = [[1, 1], [2, 1], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4], [5, 4]]
    block_size=32
    for i in mat_sizes:
        for j in mat_sizes:
            if i>=j:
                mat1_size.append([i, j])
                mat2_size.append([j, i])
                mat3_size.append([i, i])
            
    for i in range(0, len(n_proc)):
        for j in range(0, len(mat1_size)):
            
            #With 1 process load greater than that takes an eternity and it's not worth measuring performance
            if n_proc[i] == 1 and mat1_size[j][0]*mat1_size[j][1]*mat2_size[j][1] >= 34359738368:
                continue
            curr_cmd=command.format(n_proc[i], pg_size[i][0], pg_size[i][1], block_size, mat1_size[j][0], mat1_size[j][1], mat1_size[j][0], mat1_size[j][1], mat2_size[j][0], mat2_size[j][1], mat2_size[j][0], mat2_size[j][1], mat3_size[j][0], mat3_size[j][1], mat3_size[j][0], mat3_size[j][1])
            print("Launching ")
            print(curr_cmd)
            # Launch the C program using subprocess
            #process = subprocess.Popen(curr_cmd, shell=True)
            # Wait for the process to finish
           # process.wait()