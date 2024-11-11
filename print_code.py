import os

def print_python_files():
    # Loop through all files in the current directory
    for filename in os.listdir('.'):
        # Check if the file is a Python file
        if filename.endswith('.py'):
            print(f"\n{'='*40}\nFile: {filename}\n{'='*40}")
            # Open and read the file contents
            with open(filename, 'r') as file:
                contents = file.read()
                print(contents)

if __name__ == "__main__":
    print_python_files()