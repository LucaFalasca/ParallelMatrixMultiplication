import subprocess

if __name__ == "__main__":
    a=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
    
    for i in a:
        for j in a:
            if(i>=j):
                command="make runt row1="+str(i)+" col1="+str(j) + " row2="+str(j)+" col2="+str(i)
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                # Print the output
                print("Command Output:", result.stdout)

                # Print the error, if any
                if result.stderr:
                    print("Error:", result.stderr)