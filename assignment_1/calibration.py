import cv2
import numpy as np


def calibrate_camera(
    all_corners: list,
    img_shape: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size=[9,6]
)-> tuple[
    float,
    cv2.typing.MatLike,
    cv2.typing.MatLike,
    tuple[cv2.typing.MatLike],
    tuple[cv2.typing.MatLike]
]:
    """
    TODO
    """
    # Create real-world coordinates, starting at 0,0,0 for the top-left
    # point, taking steps of 1 for each new intersection.
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    real_points = [objp.copy() for _ in range(len(all_corners))]

    return cv2.calibrateCamera(
        real_points,
        all_corners,
        img_shape[::-1],
        None,
        None
    )
