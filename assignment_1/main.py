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
from display_objects import display_axis_cube, display_axis_cube_video

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
    # all_corners, img_shape = detect_corners(
    #     "assignment_1/data/all/", 
    #     output_folder,
    #     PATTERN_SIZE
    # )
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
    # _, mtx, dist, _, _ = calibrate_camera(
    #     all_corners, 
    #     img_shape, 
    #     PATTERN_SIZE, 
    #     SQUARE_SIZE
    # )

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
    #                          CHOICE TASK 2                           #
    #  Iterative detection and rejection of low quality input images.  #
    ####################################################################
    all_img_re_proj_err, _, _, _, _ = calibrate_camera(
        all_corners, 
        img_shape, 
        PATTERN_SIZE, 
        SQUARE_SIZE
    )
    image_validity_flags, (subset_img_re_proj_err, mtx, dist, _, _) = \
        check_image_calibration_contribution(
            all_img_re_proj_err, 
            all_corners, 
            img_shape, 
            PATTERN_SIZE, 
            SQUARE_SIZE
        )
    print(image_validity_flags)
    print(
        f"Calibration error with all images: {all_img_re_proj_err:.3f}\n"
        "Calibration error with only positively contributing images: "
        f"{subset_img_re_proj_err:.3f}"
    )


    ####################################################################
    #                          CHOICE TASK 5                           #
    #  Enhance the input images before attempting to detect corners.   #
    ####################################################################
    # img = cv2.imread("assignment_1/data/test.jpg")

    # adjusted_image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # adjusted_image = clahe.apply(gray)
    # # adjusted_image = cv2.medianBlur(adjusted_image, 5)


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    # sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # scale = 2.0
    # adjusted_image = cv2.resize(sharp, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # gamma = 1.5  # try 0.5 - 2.5
    # invGamma = 1.0 / gamma
    # table = np.array([(i / 255.0) ** invGamma * 255
    #                 for i in np.arange(256)]).astype("uint8")
    # adjusted_image = cv2.LUT(adjusted_image, table)

    # adjusted_image = cv2.GaussianBlur(adjusted_image, (5,5), 0)

    # gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    # kernel = np.array([[0,-1,0],
    #                 [-1,5,-1],
    #                 [0,-1,0]])
    # adjusted_image = cv2.filter2D(gray, -1, kernel)


    


    # detect_corners(img, output_folder="test_enhanced/")
    # detect_corners("assignment_1/data/mix/img_23.jpg", output_folder="test_raw/")

if __name__ == "__main__":
    main()
