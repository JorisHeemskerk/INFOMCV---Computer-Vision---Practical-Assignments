import cv2
import os
import re
import numpy as np


def detect_corners(
    source: str | cv2.typing.MatLike,
    output_folder: str | None=None,
    pattern_size: cv2.typing.Size=[9,6]
)-> tuple[list[list[tuple[float, float]]], cv2.typing.MatLike]:
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
    
    :param source: Folder in which input images are saved. If this
        source is a folder, all images will be read, if it instead is a
        file, only that file will be read, if it is an image, the image
        will be used.
    :type source: str | MatLike
    :param output_folder: Folder to which output is saved. The corners
        will be saved in a subdirectory called "corners". And the corner
        points will also be saved in a subdirectory called "data". If
        output_folder is None, no data will be saved.
    :type output_folder: str | None
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :return: List containing all the found corners of all provided 
        images, along with image dimensions
    :rtype: tuple[list[list[tuple[float, float]]], cv2.typing.MatLike]
    """
    sources = [source]
    if type(source) == str:
        if os.path.isdir(source):
            sources = sorted([
                source + filename for filename in os.listdir(source) \
                    if filename[0] != "."
            ], key=lambda s: int(re.search(r'\d+', s).group()))
    if output_folder:
        os.makedirs(output_folder + "corners/")

    corners_all_images = []
    for i, source in enumerate(sources):
        success, corners, img = automatic_corner_detector(
            source, 
            pattern_size
        )
        while success == 0:
            print(
                f"Corners were not automatically or fully manually detected in"
                f" image {source if type(source) == str else ''}.\nPlease "
                "manually click on the four corners and then close the image."
            )
            success, corners, img = manual_corner_selector(
                source, 
                pattern_size
            )
        # Correct corners by looking at surrounding pixels.
        corners_corrected = cv2.cornerSubPix(
            cv2.cvtColor(
                cv2.imread(source, 1) if type(source) == str else source, 
                cv2.COLOR_BGR2GRAY
            ),
            corners,
            (11,11), # Search window.
            (-1,-1), # No dead region.
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    
        corners_all_images.append(corners_corrected)
        if output_folder:
            filename = source.split("/")[-1] if type(source) == str \
                else f"source_{i}.jpg"
            print(filename)
            cv2.imwrite(output_folder + "corners/" + filename, img)

    if output_folder:
        os.mkdir(output_folder + "data/")
        np.save(
            output_folder + "data/corners_all_images.npy", 
            np.array(corners_all_images)
        )
        np.save(
            output_folder + "data/img_shape.npy", 
            np.array(img.shape[:2])
        )
    return corners_all_images, img.shape[:2]

def automatic_corner_detector(
    source: str | cv2.typing.MatLike, 
    pattern_size: cv2.typing.Size=[9,6]
)-> tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Automatically tries to detect corners on a chess board.
    Returns if succeeded, along with the image containing the rendered
    corners (if not successful, returns the raw image).
    
    :param source: The path to the input image or an actual image.
    :type source: str | cv2.typing.MatLike
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :return: A boolean witch reflects if the operation failed or not,
        the corners that were detected, along with the image that has 
        the rendered corners on it.
    :rtype: tuple[bool, MatLike, MatLike]
    """
    if type(source) == str:
        img = cv2.imread(source, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray, pattern_size)
        # success, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    else:
        img = source.copy()
        success, corners = cv2.findChessboardCorners(img, pattern_size)
        # success, corners = cv2.findChessboardCornersSB(img, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)

    if success != 0:
        res_img = cv2.drawChessboardCorners(img, pattern_size, corners, True)
        return success, corners, res_img
    else:
        return success, [], img

def manual_corner_selector(
    source: str | cv2.typing.MatLike, 
    pattern_size: cv2.typing.Size=[9,6]
)-> tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Manually tries to find the corners on a chess board. The user needs
    to left-click on the four corners (1 in from the outermost). For
    more accuracy the user can right-click on the image to open a
    separate window with a zoomed in area. These corners are used to
    calculate all the corners of the chess board, using `find_corners`.
    Returns if succeeded, along with the image containing the rendered
    corners (if not successful, returns the raw image).

    Code based on geeksforgeeks tutorial `Displaying the coordinates of 
    the points clicked on the image using Python-OpenCV`. Link: 
    https://www.geeksforgeeks.org/python/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    
    :param source: The path to the input image or an actual image.
    :type source: str | cv2.typing.MatLike
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
        params: int,
        flags: int,
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
        :param params: Optional parameters to pass to the function. This
            parameter is not used in this function.
        :type params: int
        :param flags: One of the cv::MouseEventFlags constants. This
            parameter is not used in this function.
        :type flags: int
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
            cv2.setWindowProperty("zoom", cv2.WND_PROP_TOPMOST, 1)

    image_corners = []

    if type(source) == str:
        img = cv2.imread(source, 1)
        zoom = cv2.imread(source, 1)
    else:
        img = source.copy()
        zoom = source.copy()
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 1, [], img
    

    if len(image_corners)==4:
        corners = find_corners(image_corners, pattern_size)
        res_img = cv2.drawChessboardCorners(zoom, pattern_size, corners, True)
        return 1, corners, res_img
    else:
        print("The correct amount of corners `4` was not provided")
        return 0, [], img

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

    upper_corners = np.linspace(
        image_corners[0],
        image_corners[1],
        pattern_size[0]
    )
    lower_corners = np.linspace(
        image_corners[2],
        image_corners[3],
        pattern_size[0]
    )
    left_corners = np.linspace(
        image_corners[0],
        image_corners[2],
        pattern_size[1]
    )
    right_corners = np.linspace(
        image_corners[1],
        image_corners[3],
        pattern_size[1]
    )

    corners = []
    for left, right in zip(left_corners, right_corners):
        for upper, lower in zip(lower_corners, upper_corners):
            corners.append([
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
            ])
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
        raise ArithmeticError("Lines are parallel and do not intersect")
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
    
    return x, y

