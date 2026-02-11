"""
This file allows one to rename long data filenames created by your
camera (e.g., the windows camera app or your phone's camera) to 
a shorter and more usable format.
"""

import os
import re


def rename_data_sequentially(folder: str, prefix: str)-> None:
    """
    Rename all data in provided folder to the format:
    `{prefix}{incremental iterator}.{original extension}`.
    Ignores hidden files.
    
    :param folder: Path to folder where the data is found
    :type folder: str
    :param prefix: prefix to put in front of incremental iterator.
    :type prefix: str
    """
    # read and sort the filenames, exclude hidden files
    filenames = sorted([
        filename for filename in os.listdir(folder) if filename[0] != "."
    ], key=lambda s: int(re.search(r'\d+', s).group()))
    for i, filename in enumerate(filenames):
        if filename[0] == ".":
            continue
        os.rename(
            folder + filename, 
            folder + prefix + str(i) + "." + filename.split(".")[-1]
        )
    
def main():
    rename_data_sequentially(folder= "assignment_1/data/", prefix="img_")

if __name__ == "__main__":
    main()
