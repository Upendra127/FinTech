import os
from tqdm import tqdm
import shutil
import subprocess

def delete_files_in_directory(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        filepath = os.path.join(directory, filename)
        try:
            # Check if the path is a file
            if os.path.isfile(filepath):
                # Delete the file
                os.remove(filepath)
                print(f"Deleted file: {filepath}")
            elif os.path.isdir(filepath):
                # Delete the directory and its contents
                shutil.rmtree(filepath)
                print(f"Deleted directory: {filepath}")
        except Exception as e:
            # Handle any errors that occur during file deletion
            print(f"Error deleting file {filepath}: {e}")

# Example usage:
directories = []
directories.append("./sec-edgar-filings")
directories.append("./data/form10k.parsed")
directories.append("./data/mda")
directories.append("./data/form10k")
# directories.append("./static")
for dir in tqdm(directories):
    delete_files_in_directory(dir)
