import cv2

import numpy as np


def draw_axis(
    img: cv2.typing.MatLike,
    rvec: cv2.typing.MatLike,
    tvec: cv2.typing.MatLike,
    mtx: cv2.typing.MatLike,
    dist: cv2.typing.MatLike,
    square_size: float
)-> None:
    """
    Draw X, Y and Z axis on top of the chessboard from the corner that
    is defined as (0, 0, 0) in world coordinates onto a 2D image. 

    :param img: The image.
    :type img: MatLike
    :param rvec: Rotation vector.
    :type rvec: MatLike | None
    :param tvec: Translation vector.
    :type tvec: MatLike | None
    :param mtx: Calibration matrix.
    :type mtx: MatLike
    :param dist: Distance matrix.
    :type dist: MatLike
    :param square_size: The length of a single chessboard square.
    :type square_size: float
    """
    axis_points = np.array([
        [0, 0, 0],
        [5, 0, 0],
        [0, 5, 0],
        [0, 0, -5]
    ], dtype=np.float32) * square_size

    points, _ = cv2.projectPoints(
        axis_points,
        rvec,
        tvec,
        mtx,
        dist
    )
    points = points.reshape(-1, 2)

    origin = [int(points[0][0]), int(points[0][1])]
    x_axis = [int(points[1][0]), int(points[1][1])]
    y_axis = [int(points[2][0]), int(points[2][1])]
    z_axis = [int(points[3][0]), int(points[3][1])]

    cv2.arrowedLine(img, origin, x_axis, (0,0,255))
    cv2.arrowedLine(img, origin, y_axis, (0,255,0))
    cv2.arrowedLine(img, origin, z_axis, (255,0,0))
