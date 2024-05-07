import subprocess

def execute_cmd(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Print the output
    print("Command Output:", result.stdout)

    # Print the error, if any
    if result.stderr:
        print("Error:", result.stderr)
    

if __name__ == "__main__":
    a = [50, 100, 500, 1000, 2048, 3000, 4096, 6000, 8192, 9000, 11000]
    b = [32, 64, 128 ,156]
    
    for i in a:
        command="make runt row1="+str(i)+" col1="+str(i) + " row2="+str(i)+" col2="+str(i)
        execute_cmd(command)
        for j in b:
            command="make runt row1="+str(i)+" col1="+str(j) + " row2="+str(j)+" col2="+str(i)
            execute_cmd(command)