import cv2
import os
import re
import datetime
import numpy as np

from calibration import \
    calibrate_camera, \
    get_rvec_tvec, \
    check_image_calibration_contribution
from detect_corners import detect_corners
from display_objects import \
    display_axis_cube, \
    display_axis_cube_video, \
    plot_calibration_cameras

# Pattern size of chessboard
PATTERN_SIZE = [9,6]
# Length of a single square on chessboard (in meters)
SQUARE_SIZE =0.024


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
    all_corners, img_shape = detect_corners(
        "assignment_1/data/all/", 
        output_folder,
        PATTERN_SIZE
    )
    #
    # 2. Only 5 correct + 5 manual images
    # all_corners, img_shape = detect_corners(
    #     "assignment_1/data/mix/", 
    #     output_folder,
    #     PATTERN_SIZE
    # )
    #
    # 3. Only 5 correct (from 2)
    # all_corners, img_shape = detect_corners(
    #     "assignment_1/data/auto/", 
    #     output_folder,
    #     PATTERN_SIZE
    # )
    #
    # 4. Import the corners from the latest previous run.
    # all_corners = np.load(previous_run_data_dir + "corners_all_images.npy")
    # img_shape   = np.load(previous_run_data_dir + "img_shape.npy")
    # 
    # 5. Import the corners from a manual directory.
    # previous_run_data_dir = "insert_path_here"
    # all_corners = np.load(previous_run_data_dir + "corners_all_images.npy")
    # img_shape   = np.load(previous_run_data_dir + "img_shape.npy")


    ####################################################################
    #                      Calibrate the camera.                       #
    ####################################################################
    _, mtx, dist, rvecs, tvecs = calibrate_camera(
        all_corners, 
        img_shape, 
        PATTERN_SIZE, 
        SQUARE_SIZE
    )

    ####################################################################
    #           Display the axis and cube on the test image.           #
    ####################################################################
    # display_axis_cube(
    #     "assignment_1/data/test/img_25.jpg",
    #     *get_rvec_tvec(
    #         "assignment_1/data/test/img_25.jpg", 
    #         mtx, 
    #         dist, 
    #         PATTERN_SIZE,
    #         SQUARE_SIZE
    #     ),
    #     mtx,
    #     dist,
    #     "image",
    #     True
    # )

    
    ####################################################################
    #                          CHOICE TASK 5                           #
    #  Enhance the input images before attempting to detect corners.   #
    ####################################################################
    # for i in [1, 2, 15, 18, 23]: 
    #     print(f"Attempting to calibrate image {i}.")

    #     img = cv2.imread(f"assignment_1/data/all/img_{i}.jpg")
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #     _, adjusted_image = cv2.threshold(
    #         gray,
    #         200, # Threshold value of 200
    #         255,
    #         cv2.THRESH_BINARY_INV
    #     )

    #     # More exhaustive way to find chessboard corners, even though it
    #     # is substantially slower.
    #     success, _ = cv2.findChessboardCornersSB(
    #         adjusted_image, 
    #         PATTERN_SIZE, 
    #         flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    #     )

    #     if success:
    #         print(f" - Calibration that failed before is now working!")
    #     else:
    #         print(f" - Calibration still does not work.")

    ####################################################################
    #                          CHOICE TASK 2                           #
    #  Iterative detection and rejection of low quality input images.  #
    ####################################################################
    # all_img_re_proj_err, _, _, _, _ = calibrate_camera(
    #     all_corners, 
    #     img_shape, 
    #     PATTERN_SIZE, 
    #     SQUARE_SIZE
    # )
    # image_validity_flags, (subset_img_re_proj_err, mtx, dist, _, _) = \
    #     check_image_calibration_contribution(
    #         all_img_re_proj_err, 
    #         all_corners, 
    #         img_shape, 
    #         PATTERN_SIZE, 
    #         SQUARE_SIZE
    #     )
    # print(
    #     f"Calibration error with all images: {all_img_re_proj_err:.3f}\n"
    #     "Calibration error with only positively contributing images: "
    #     f"{subset_img_re_proj_err:.3f}\nA total of "
    #     f"{len(image_validity_flags) - sum(image_validity_flags)} images "
    #     f"have been removed.\n{image_validity_flags=}"
    # )

    ####################################################################
    #                          CHOICE TASK 1                           #
    #          Display the axis and cube on the live webcam.           #
    ####################################################################
    # display_axis_cube_video(
    #     cv2.VideoCapture(0), 
    #     mtx, 
    #     dist, 
    #     "video", 
    #     PATTERN_SIZE,
    #     SQUARE_SIZE
    # )

    ####################################################################
    #                          CHOICE TASK 6                           #
    #  3D plot the locations of the camera relative to the chessboard  #
    ####################################################################
    plot_calibration_cameras(rvecs, tvecs)

if __name__ == "__main__":
    main()
