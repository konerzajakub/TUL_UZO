import os
import subprocess

def launch_all_python_files(directory):
    filenames = os.listdir(directory)
    pyfiles = []
    
    for filename in filenames:
        if filename.endswith(".py") and filename != os.path.basename(__file__):
            pyfiles.append(filename)
    
    pyfiles.sort()

    for filename in pyfiles:
            print(f'Launching {filename}')
            filepath = os.path.join(directory, filename)
            subprocess.run(["python3", filepath])

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    launch_all_python_files(current_directory)