import cv2
import os
import re
import datetime

from autmatic_corners import automatic_corner_detector


def detect_corners(folder: str, output_folder: str)-> None:
    # read and sort the filenames, exclude hidden files
    filenames = sorted([
        filename for filename in os.listdir(folder) if filename[0] != "."
    ], key=lambda s: int(re.search(r'\d+', s).group()))
    for filename in filenames:
        success, corners, img = automatic_corner_detector(folder + filename)
        if success == 0:
            print(f"corners were not detected in image {filename}.")
        cv2.imwrite(output_folder + filename, img)
    

def main()-> None:
    output_folder = "assignment_1/output/run_" + \
        f"{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}/"
    os.mkdir(output_folder)
    detect_corners("assignment_1/data/", output_folder)

if __name__ == "__main__":
    main()
