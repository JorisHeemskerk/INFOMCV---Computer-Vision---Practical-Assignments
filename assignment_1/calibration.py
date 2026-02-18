import cv2
import numpy as np

from tqdm import tqdm

from detect_corners import detect_corners, automatic_corner_detector


def calibrate_camera(
    all_corners: list[list[tuple[float, float]]],
    img_shape: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size=[9,6],
    square_size: float=0.024
)-> tuple[
    float,
    cv2.typing.MatLike,
    cv2.typing.MatLike,
    tuple[cv2.typing.MatLike],
    tuple[cv2.typing.MatLike]
]:
    """
    Calibrate a camera based on a set of detected corners from chess 
    boards.

    First, this function establishes real-world xyz coordinates. These
    are used together with the detected `all_corners`, in order to
    estimate the camera properties.

    :param all_corners: Container for all the corners of all the images.
    :type all_corners: list[list[tuple[float, float]]]
    :param img_shape: Shape (x,y) of the original images in pixels.
    :type img_shape: cv2.typing.MatLike
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :param square_size: The length of a single chessboard square.
    :type square_size: float
    :return: Ret, mtx, dist, rvecs, tvecs.
    :rtype: tuple[float, MatLike, MatLike, tuple[MatLike], tuple[MatLike]]
    """
    # Create real-world coordinates, starting at 0,0,0 for the top-left
    # point, taking steps of 1 for each new intersection.
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size
    real_points = [objp.copy() for _ in range(len(all_corners))]

    return cv2.calibrateCamera(
        real_points,
        all_corners,
        img_shape[::-1],
        None,
        None
    )

def get_rvec_tvec(
    source: str | cv2.typing.MatLike,
    mtx: cv2.typing.MatLike,
    dist: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size=[9,6],
    square_size: float=0.024
)-> tuple[cv2.typing.MatLike | None, cv2.typing.MatLike | None]:
    """
    Get the rotation and translation vectors form an image.

    Calculate the rotation and translation vectors from an image using
    the calibrated matrix and distances. This function detects the
    chessboard corners in an image and calculates the vectors. If the
    provided image comes from a file, the user may be prompted to 
    manually detect the corners. If an already loaded image is passed,
    and no corners can automatically be detected, the function will
    return two None vectors.

    :param source: Path to input image or an actual input image
    :type source: str | cv2.typing.MatLike
    :param mtx: The camera intrinsics matrix
    :type mtx: cv2.typing.MatLike
    :param dist: The distortion coefficients
    :type dist: cv2.typing.MatLike
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :param square_size: The length of a single chessboard square.
    :type square_size: float
    :return: The rotation vector and translation vector of the input 
        image
    :rtype: tuple[MatLike, MatLike]
    """
    if type(source) == str:
        img = cv2.imread(source, 1)
        corners, _ = detect_corners(source, None, pattern_size)
        # Extract the first and only corners element.
        corners = corners[0]
    else:
        img = source.copy()
        success, corners, _ = automatic_corner_detector(img, pattern_size)
        if success == 0:
            return None, None

    corners_corrected = cv2.cornerSubPix(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        corners,
        (11,11),
        (-1,-1), # No dead region.
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size

    _, rvec, tvec = cv2.solvePnP(objp, corners_corrected, mtx, dist)
    return rvec, tvec

def check_image_calibration_contribution(
    all_img_re_proj_err: float,
    all_corners: np.ndarray,
    img_shape: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size=[9,6],
    square_size: float=0.024
)-> tuple[
    np.ndarray, 
    tuple[
        float,
        cv2.typing.MatLike,
        cv2.typing.MatLike,
        tuple[cv2.typing.MatLike],
        tuple[cv2.typing.MatLike]
    ]]:
    """

    
    :param all_img_re_proj_err: Description
    :type all_img_re_proj_err: float
    :param all_corners: Container for all the corners of all the images.
    :type all_corners: np.ndarray
    :param img_shape: 
    :type img_shape: MatLike
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :param square_size: The length of a single chessboard square.
    :type square_size: float
    :return: List of flags with 1 when image has positive 
        contribution and 0 if not, along with the calibration 
        elements of the best combination of images.
    :rtype: tuple[
        np.ndarray, 
        tuple[
            float, 
            MatLike,
            MatLike,
            tuple[MatLike],
            tuple[MatLike]
        ]]
    """
    image_validity_flags = np.zeros(len(all_corners), dtype=bool)
    for i in tqdm(
        range(len(all_corners)), 
        desc="Leaving out 1 image and checking how calibration changes"
    ):
        mask = np.ones(len(all_corners), dtype=bool)
        mask[i] = 0
        re_proj_err, _, _, _, _ = calibrate_camera(
            all_corners[mask], 
            img_shape, 
            pattern_size, 
            square_size
        )
        # Check if the image has a negative impact on the calibration.
        if re_proj_err > all_img_re_proj_err:
            image_validity_flags[i] = 1
    return image_validity_flags, calibrate_camera(
            all_corners[image_validity_flags], 
            img_shape, 
            pattern_size, 
            square_size
        )
