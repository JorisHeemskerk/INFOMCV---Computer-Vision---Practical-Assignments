import cv2
import numpy as np

from detect_corners import detect_corners
from display_objects import draw_axis, draw_cube


def process_image(
    img_path: str,
    mtx: cv2.typing.MatLike,
    dist: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size=[9,6]
)-> None:
    """
    TODO
    """
    corners, _ = detect_corners(img_path)
    # Extract the first and only corners element.
    corners = corners[0]

    corners_corrected = cv2.cornerSubPix(
        cv2.cvtColor(cv2.imread(img_path, 1), cv2.COLOR_BGR2GRAY),
        corners,
        (11,11),
        (-1,-1), # No dead region.
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)

    a, rvec, tvec = cv2.solvePnP(objp, corners_corrected, mtx, dist)

    img = cv2.imread(img_path, 1)
    draw_axis(img, rvec, tvec, mtx, dist)
    draw_cube(img, rvec, tvec, mtx, dist)

    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
