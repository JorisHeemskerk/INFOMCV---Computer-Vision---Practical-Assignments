import cv2


def automatic_corner_detector(
    img_path: str, 
    pattern_size: cv2.typing.Size=[9,6]
)-> tuple[bool, cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Automatically tries to detect corners on a chess board.
    Returns if succeeded, along with the image containing the rendered
    corners (if not successful, returns the raw image).
    
    :param img_path: The path to the input image.
    :type img_path: str
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :return: A boolean witch reflects if the operation failed or not,
        the corners that were detected, along with the image that has 
        the rendered corners on it.
    :rtype: tuple[bool, MatLike, MatLike]
    """
    img = cv2.imread(img_path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, pattern_size)
    if success != 0:
        res_img = cv2.drawChessboardCorners(img, pattern_size, corners, True)
        return success, corners, res_img
    else:
        return success, [], img
