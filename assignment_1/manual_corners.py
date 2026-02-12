"""
This file allows one to select points on an image for manual calibration.

Code based on geeksforgeeks tutorial `Displaying the coordinates of the 
points clicked on the image using Python-OpenCV`
link: https://www.geeksforgeeks.org/python/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
"""

import cv2
import numpy as np


def manual_corner_selector(
    img_path: str,
    pattern_size: cv2.typing.Size=[9,6]
)-> tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Manually tries to find the corners on a chess board. The user needs
    to left-click on the four outermost corners (For more accuracy the
    user can right-click on the image to open a separate window with
    a zoomed in area). These corners are used to calculate all the
    corners of the chess board.
    Returns if succeeded, along with the image containing the rendered
    corners (if not successful, returns the raw image).
    
    :param img_path: The path to the input image.
    :type img_path: str
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :return: A boolean witch reflects if the operation failed or not,
        The corners that were detected, along with the image that has 
        the rendered corners on it.
    :rtype: tuple[bool, MatLike, MatLike]
    """
    def click_event(
        event: int,
        x: int,
        y: int,
        flags: int,
        params: any | None
    )-> None:
        """
        Mouse interaction with an opened cv2 window event handler.
        Allows the coordinates of the first four left clicks to be
        added to the image_corners list. Right clicking gives a zoomed
        in version of the window that's zoomed in on.

        :param event: One of the cv::MouseEventTypes constants
        :type event: int
        :param x: The x-coordinate of the mouse event.
        :type x: int
        :param y: The y-coordinate of the mouse event.
        :type y: int
        :param flags: One of the cv::MouseEventFlags constants.
        :type flags: int
        :param params: Optional parameters.
        :type params: any | None
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(image_corners) < 4:
            image_corners.append((x, y))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"{x},{y}", (x, y), font, 1, (255, 0, 0), 2)
            cv2.imshow('image', img)

        if event == cv2.EVENT_RBUTTONDOWN:
            zoom_display = zoom[y-10:y+10, x-10:x+10].copy()
            cv2.circle(zoom_display, (10, 10), 0, (255, 0, 0))
            zoom_display = cv2.resize(zoom_display, (200, 200))
            cv2.imshow('zoom', zoom_display)

    image_corners = []

    img = cv2.imread(img_path, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    zoom = cv2.imread(img_path, 1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(image_corners)==4:
        corners = find_corners(image_corners, pattern_size)

        res_img = cv2.drawChessboardCorners(img, pattern_size, corners, True)

        return True, corners, res_img
    else:
        print("The correct amount of corners `4` was not provided")
        return False, [], img


def find_corners(
    image_corners: list[tuple[int, int]],
    pattern_size: cv2.typing.Size=[9,6]
)-> cv2.typing.MatLike:
    """
    Finds the corners on a chess board given the outer image corners of
    the chess board. This is done by finding which of the corners in
    image_corners is the corner in the top left(TL), top right(TR),
    bottom left(BL) and bottom right(BR). These corners are then sorted
    in this order [TR, TL, BR, BL]. Using linear interpolation the
    points between these corners according to the pattern_size are
    found. Afterwards these outer line corners are used to find the rest
    of the corners on the grid by using the intersection between lines
    drawn between the corresponding outer line corners.

    Usage of linspace (linear interpolation) was found from:
    https://stackoverflow.com/questions/47443037/equidistant-points-between-two-points

    :param image_corners: 
    :type image_corners: list[tuple[int, int]]
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :return: The corners that were found.
    :rtype: MatLike
    """
    corners_sorted = sorted(image_corners, key=lambda x: x[1], reverse=True)

    top = sorted(corners_sorted[:2], key=lambda x: x[0], reverse=True)
    bottom = sorted(corners_sorted[2:], key=lambda x: x[0], reverse=True)
    
    image_corners = top + bottom

    upper_corners = np.linspace(image_corners[0], image_corners[1], pattern_size[0])
    lower_corners = np.linspace(image_corners[2], image_corners[3], pattern_size[0])
    left_corners = np.linspace(image_corners[0], image_corners[2], pattern_size[1])
    right_corners = np.linspace(image_corners[1], image_corners[3], pattern_size[1])

    corners = []
    for left, right in zip(left_corners, right_corners):
        for upper, lower in zip(lower_corners, upper_corners):
                  
            corners.append(
                [
                    line_intersect(
                        upper[0],
                        upper[1],
                        lower[0],
                        lower[1],
                        left[0],
                        left[1],
                        right[0],
                        right[1]
                    )
                ]
            )
    return np.asarray(corners, dtype=np.float32)


def line_intersect(
    Ax1: float,
    Ay1: float,
    Ax2: float,
    Ay2: float,
    Bx1: float,
    By1: float,
    Bx2: float,
    By2: float
)-> tuple[float, float] | None:
    """
    Finds the (x, y) point where the there is an intersection between
    the line that goes between (Ax1, Ay1) and (Ax2, Ay2) and the line 
    that goes between (Bx1, By1) and (Bx2, By2).
    Code found from: 
    https://rosettacode.org/wiki/Find_the_intersection_of_two_lines#Python

    :param Ax1: X coordinate of the first point on line A.
    :type Ax1: float
    :param Ay1: Y coordinate of the first point on line A.
    :type Ay1: float
    :param Ax2: X coordinate of the second point on line A.
    :type Ax2: float
    :param Ay2: Y coordinate of the second point on line A.
    :type Ay2: float
    :param Bx1: X coordinate of the first point on line B.
    :type Bx1: float
    :param By1: Y coordinate of the first point on line B.
    :type By1: float
    :param Bx2: X coordinate of the second point on line B.
    :type Bx2: float
    :param By2: Y coordinate of the second point on line B.
    :type By2: float
    :return: The point that was found on the intersection between the
        lines A and B.
    :rtype: tuple[float, float]
    """
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    if d:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
    else:
        return
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
    
    return x, y

