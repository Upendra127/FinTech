import os
import shutil
from pathlib import Path
from tqdm import tqdm
import subprocess

# Define a custom sorting function
def sort_key(directory):
    # Extract the numeric part from the directory name
    numeric_part = directory.split('-')[1]  # Assuming the directory name is like '0000891020-94-000175'
    return int(numeric_part)

def sorter(ticker):
    # Parent directory containing the folders
    sec_filings = Path('/Users/chinu/Desktop/FinTech copy/sec-edgar-filings')
    parent_dir = os.path.join(sec_filings,ticker,"10-K")

    trg = Path("/Users/chinu/Desktop/FinTech copy/data/form10k")
    # List all directories in the parent directory
    directories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # Sort directories based on the custom key function
    sorted_directories = sorted(directories, key=sort_key)

    # Print sorted directories
    print("Sorted Directories:")
    for directory in tqdm(sorted_directories):
        dir = Path(os.path.join(parent_dir,directory))
        print(dir)
        if dir.is_dir():
            # Get the first file inside the directory
            print("hih")
            file_inside = list(dir.iterdir())
            if file_inside:
                src_file = file_inside[0]
                dest_file = trg / f"{directory.split('-')[1]}.txt"
                print(dest_file)
                shutil.move(src_file, dest_file)