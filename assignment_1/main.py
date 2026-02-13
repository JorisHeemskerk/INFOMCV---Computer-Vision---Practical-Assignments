import cv2
import os
import re
import datetime
import numpy as np

from autmatic_corners import automatic_corner_detector
from manual_corners import manual_corner_selector


def detect_corners(folder: str, output_folder: str)-> None:
    """
    Detect the corners of the chessboards in every image in provided
    folder. 
    
    Function first attempts to select the corners automatically. If this
    fails, the user is prompted to select the four outer corners by hand
    and then close the window(s). A user can right click to show where
    a corner would be placed, in a separate window, as well as left 
    click to place a definitive marker.
    
    :param folder: Folder in which input images are saved.
    :type folder: str
    :param output_folder: Folder to which output is saved. The corners
    will be saved in a subdirectory called "corners". And the corner
    points will also be saved in a subdirectory called "data"
    :type output_folder: str
    """
    # threedpoints = []
    # twodpoints = []

    # objectp3d = np.zeros((1, 9 * 6, 3), np.float32)
    # objectp3d[0, :, :2] = np.mgrid[0:9, 6].T.reshape(-1, 2)
    # Read and sort the filenames, exclude hidden files.
    filenames = sorted([
        filename for filename in os.listdir(folder) if filename[0] != "."
    ], key=lambda s: int(re.search(r'\d+', s).group()))
    os.mkdir(output_folder + "corners/")

    corners_all_images = []
    for filename in filenames:
        success, corners, img = automatic_corner_detector(folder + filename)
        while success == 0:
            print(
                f"Corners were not automatically or fully manually detected in"
                f" image {filename}.\nPlease manually click on the four "
                "corners and then close the image."
            )
            success, corners, img = manual_corner_selector(folder + filename)

    #     threedpoints.append(objectp3d)
    #     # tutorial refines pixel coordinates for given 2d points with
    #     # cv2.cornerSubPix
    #     twodpoints.append(corners)
    #     cv2.imwrite(output_folder + filename, img)
    # return threedpoints, twodpoints
    
        corners_all_images.append(corners)
        cv2.imwrite(output_folder + "corners/" + filename, img)

    os.mkdir(output_folder + "data/")
    np.save(
        output_folder + "data/corners_all_images.npy", 
        np.array(corners_all_images)
    )

def main()-> None:
    # Create output folder for the finished images.
    output_folder = "assignment_1/output/run_" + \
        f"{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}/"
    os.mkdir(output_folder)
    # threedpoints, twodpoints = detect_corners("assignment_1/data/", output_folder)

    # ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    #     threedpoints, twodpoints, grayColor.shape[::-1], None, None
    # )

    detect_corners("assignment_1/data/", output_folder)

if __name__ == "__main__":
    main()
