import cv2
import os
import re
import datetime
import numpy as np

from autmatic_corners import automatic_corner_detector
from manual_corners import manual_corner_selector


def detect_corners(folder: str, output_folder: str)-> None:
    threedpoints = []
    twodpoints = []

    objectp3d = np.zeros((1, 9 * 6, 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:9, 6].T.reshape(-1, 2)

    # read and sort the filenames, exclude hidden files
    filenames = sorted([
        filename for filename in os.listdir(folder) if filename[0] != "."
    ], key=lambda s: int(re.search(r'\d+', s).group()))
    for filename in filenames:
        success, corners, img = automatic_corner_detector(folder + filename)
        if success == 0:
            print(f"corners were not auto detected in image {filename}.")
            success, corners, img = manual_corner_selector(folder + filename)

        threedpoints.append(objectp3d)
        # tutorial refines pixel coordinates for given 2d points with
        # cv2.cornerSubPix
        twodpoints.append(corners)
        cv2.imwrite(output_folder + filename, img)
    return threedpoints, twodpoints
    

def main()-> None:
    output_folder = "assignment_1/output/run_" + \
        f"{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}/"
    os.mkdir(output_folder)
    threedpoints, twodpoints = detect_corners("assignment_1/data/", output_folder)

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None
    )

if __name__ == "__main__":
    main()
