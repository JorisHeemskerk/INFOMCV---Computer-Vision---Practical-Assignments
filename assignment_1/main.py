import cv2
import os
import re
import datetime
import numpy as np

from autmatic_corners import automatic_corner_detector
from manual_corners import manual_corner_selector


def detect_corners(
    folder: str,
    output_folder: str
)-> tuple[list, cv2.typing.MatLike]:
    """
    Detect the corners of the chessboards in every image in provided
    folder. 
    
    Function first attempts to select the corners automatically. If this
    fails, the user is prompted to select the four outer corners by hand
    and then close the window(s). A user can right click to show where
    a corner would be placed, in a separate window, as well as left 
    click to place a definitive marker.
    Some code inspired by:
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    
    :param folder: Folder in which input images are saved.
    :type folder: str
    :param output_folder: Folder to which output is saved. The corners
    will be saved in a subdirectory called "corners". And the corner
    points will also be saved in a subdirectory called "data"
    :type output_folder: str
    :return: List containing all the found corners of all images in 
        folder.
    :rtype: list
    """
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
        # Correct corners by looking at surrounding pixels.
        corners_corrected = cv2.cornerSubPix(
            cv2.cvtColor(cv2.imread(folder + filename, 1), cv2.COLOR_BGR2GRAY),
            corners,
            (11,11),
            (-1,-1), # No dead region.
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    
        corners_all_images.append(corners_corrected)
        cv2.imwrite(output_folder + "corners/" + filename, img)

    os.mkdir(output_folder + "data/")
    np.save(
        output_folder + "data/corners_all_images.npy", 
        np.array(corners_all_images)
    )
    return corners_all_images, img.shape[:2]


def calibrate_camera(
    all_corners: list,
    img_shape: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size=[9,6]
)-> cv2.typing.MatLike:
    """
    """
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    real_points = [objp.copy() for _ in range(len(all_corners))]
    # print(real_points[0])
    # print(all_corners[0])
    # print(img_shape)
    # exit()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        real_points,
        all_corners,
        img_shape[::-1],
        None,
        None
    )
    print(f"{ret = }")
    print(f"{mtx = }")
    print(f"{dist = }")
    print(f"{rvecs = }")
    print(f"{tvecs = }")

def main()-> None:
    # Create output folder for the finished images.
    output_folder = "assignment_1/output/run_" + \
        f"{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}/"
    os.mkdir(output_folder)

    all_corners, img_shape = detect_corners("assignment_1/data/", output_folder)
    calibrate_camera(all_corners, img_shape)

if __name__ == "__main__":
    main()
