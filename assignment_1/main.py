import cv2
import os
import re
import datetime
import numpy as np

from calibration import calibrate_camera, get_rvec_tvec
from detect_corners import detect_corners
from display_objects import display_axis_cube, display_axis_cube_video


def main()-> None:
    # Create output folder name for the finished images.
    output_folder = "assignment_1/output/run_" + \
        f"{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}/"

    # Find and select the cached corners of the previous run if there.
    previous_run_data_dir = "assignment_1/output/" + \
        sorted(
            [
                filename for filename in os.listdir(
                    "assignment_1/output/"
                ) if filename[0] != "."
            ], 
            key=lambda s: int(re.search(r'\d+', s).group())
        )[-1] + \
        "/data/"


    ####################################################################
    # Select a data subset to run or continue on basis of previous run.#
    ####################################################################
    #
    # 1. All 20 correct + 5 manual images
    # all_corners, img_shape = detect_corners(
    #     "assignment_1/data/all/", 
    #     output_folder
    # )
    #
    # 2. Only 5 correct + 5 manual images
    # all_corners, img_shape = detect_corners(
    #     "assignment_1/data/mix/", 
    #     output_folder
    # )
    #
    # 3. Only 5 correct (from 2)
    # all_corners, img_shape = detect_corners(
    #     "assignment_1/data/auto/", 
    #     output_folder
    # )
    #
    # 4. Import the corners from the latest previous run.
    all_corners = np.load(previous_run_data_dir + "corners_all_images.npy")
    img_shape   = np.load(previous_run_data_dir + "img_shape.npy")
    # 
    # 5. Import the corners from a manual directory.
    # previous_run_data_dir = "insert_path_here"
    # all_corners = np.load(previous_run_data_dir + "corners_all_images.npy")
    # img_shape   = np.load(previous_run_data_dir + "img_shape.npy")


    ####################################################################
    #                      Calibrate the camera.                       #
    ####################################################################
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(all_corners, img_shape)

    ####################################################################
    #           Display the axis and cube on the test image.           #
    ####################################################################
    # display_axis_cube(
    #     "assignment_1/data/test/img_25.jpg",
    #     *get_rvec_tvec("assignment_1/data/test/img_25.jpg", mtx, dist),
    #     mtx,
    #     dist,
    #     "image",
    #     True
    # )

    ####################################################################
    #          Display the axis and cube on the live webcam.           #
    ####################################################################
    display_axis_cube_video(cv2.VideoCapture(0), mtx, dist, "video")

if __name__ == "__main__":
    main()
