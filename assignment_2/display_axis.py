import cv2

import numpy as np


def draw_axis(
    img: cv2.typing.MatLike,
    rvec: cv2.typing.MatLike,
    tvec: cv2.typing.MatLike,
    mtx: cv2.typing.MatLike,
    dist: cv2.typing.MatLike,
    square_size: float,
    flip_z: bool=True
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
    :param flip_z: Determines if the z axis should be flipped or not.
    :type flip_z: bool
    """
    axis_points = np.array([
        [0, 0, 0],
        [5, 0, 0],
        [0, 5, 0],
        [0, 0, 5]
    ], dtype=np.float32) * square_size

    if flip_z:
        axis_points[:, 2] *= -1

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


def draw_cube(
    img: cv2.typing.MatLike,
    rvec: cv2.typing.MatLike,
    tvec: cv2.typing.MatLike,
    mtx: cv2.typing.MatLike,
    dist: cv2.typing.MatLike,
    flip_z: bool=True
)-> None:
    """
    Draw a cube on top of the chessboard from the corner that is defined
    as (0, 0, 0) in world coordinates onto a 2D image. The cube is
    3x3x1.5 squares big.

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
    :param flip_z: Determines if the z axis should be flipped or not.
    :type flip_z: bool
    """
    # TL = (-1.2, -1.4)
    # BR = (1.8, 1.6)
    TL = (-0.15, -0.4)
    BR = (0.85, 0.6)
    z = 1.5
    cube_points = np.array([
        [TL[0], TL[1], 0],
        [BR[0], TL[1], 0],
        [TL[0], BR[1], 0],
        [BR[0], BR[1], 0],
        [TL[0], TL[1], z],
        [BR[0], TL[1], z],
        [TL[0], BR[1], z],
        [BR[0], BR[1], z]
    ], dtype=np.float32)

    if flip_z:
        cube_points[:, 2] *= -1

    points, _ = cv2.projectPoints(
        cube_points,
        rvec,
        tvec,
        mtx,
        dist
    )
    points = points.reshape(-1, 2)

    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    for start, end in edges:
        cv2.line(
            img,
            (int(points[start][0]), int(points[start][1])),
            (int(points[end][0]), int(points[end][1])),
            (230, 216, 173),
            2
        )
    cv2.circle(img, (int(points[0][0]), int(points[0][1])), 3, (0, 0, 255))
    cv2.circle(img, (int(points[3][0]), int(points[3][1])), 3, (0, 255, 0))

