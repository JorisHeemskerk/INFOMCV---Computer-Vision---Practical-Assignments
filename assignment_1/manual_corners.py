"""
This file allows one to select points on an image for manual calibration.

Code based on geeksforgeeks tutorial `Displaying the coordinates of the 
points clicked on the image using Python-OpenCV`
link: https://www.geeksforgeeks.org/python/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
"""

import cv2


def manual_corner_selector(
    img_path: str,
    pattern_size: cv2.typing.Size=[9,6]
)-> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Docstring for manual_corner_selector
    TODO
    
    :param img_path: The path to the input image.
    :type img_path: str
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :return: The corners that were detected, along with the image that
    :rtype: tuple[MatLike, MatLike]
    """
    def click_event(
        event: int,
        x: int,
        y: int,
        flags: int,
        params: any | None
    )-> None:
        """
        TODO
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(image_corners) < 4:
            print(x, y)
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
        return False, [], []


def find_corners(
    image_corners: list[tuple[int, int]],
    pattern_size: cv2.typing.Size=[9,6]
)-> cv2.typing.MatLike:
    """
    TODO
    """
    


if __name__=="__main__":
    # img = cv2.imread('assignment_1/data/img_0.jpg', 1)
    # cv2.imshow('image', img)
    # cv2.setMouseCallback('image', click_event)
    # zoom = cv2.imread('assignment_1/data/img_0.jpg', 1)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    a, b = manual_corner_selector('assignment_1/data/img_0.jpg')
    print(a)

