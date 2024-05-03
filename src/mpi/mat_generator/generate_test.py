import subprocess

def execute_cmd(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Print the output
    print("Command Output:", result.stdout)

    # Print the error, if any
    if result.stderr:
        print("Error:", result.stderr)
    

if __name__ == "__main__":
    
    for i in range(-20, 20):
        command="make runt row1="+str(4096+i)+" col1="+str(4096+i) + " row2="+str(4096+i)+" col2="+str(4096+i)
        execute_cmd(command)