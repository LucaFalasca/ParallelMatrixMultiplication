import subprocess

if __name__ == "__main__":
    command = "mpirun -n {} ./a.out {} {} {} data/matrix/bin/mat1_{}x{}.bin {} {} data/matrix/bin/mat2_{}x{}.bin {} {} data/matrix/bin/mat3_{}x{}.bin data/matrix/bin/mat3_{}x{}_check.bin"
    
    block_size=32
    
    mat1_size = []
    mat2_size = []
    mat3_size = []
    
    
    for i in range(-20,21):
        #Squared matrices
        mat1_size.append([4096+i, 4096+i])
        mat2_size.append([4096+i, 4096+i])
        mat3_size.append([4096+i, 4096+i])
        
    for j in range(0, len(mat1_size)):
        curr_cmd=command.format(20, 5, 4, block_size, mat1_size[j][0], mat1_size[j][1], mat1_size[j][0], mat1_size[j][1], mat2_size[j][0], mat2_size[j][1], mat2_size[j][0], mat2_size[j][1], mat3_size[j][0], mat3_size[j][1], mat3_size[j][0], mat3_size[j][1])
        print("Launching ")
        print(curr_cmd)
        # Launch the C program using subprocess
        process = subprocess.Popen(curr_cmd, shell=True)
        # Wait for the process to finish
        process.wait()
        
