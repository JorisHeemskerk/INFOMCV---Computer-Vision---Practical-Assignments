import cv2
import numpy as np

from calibration import get_rvec_tvec


def draw_axis(
    img: cv2.typing.MatLike,
    rvec: cv2.typing.MatLike,
    tvec: cv2.typing.MatLike,
    mtx: cv2.typing.MatLike,
    dist: cv2.typing.MatLike
)-> None:
    """
    TODO
    """
    axis_points = np.array([
        [0, 0, 0],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, -3]
    ], dtype=np.float32)

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
    square_size: float=0.024
)-> None:
    """
    TODO
    """
    cube_points = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [2, 2, 0],
        [0, 0, -2],
        [2, 0, -2],
        [0, 2, -2],
        [2, 2, -2],
        [1, 1, -2]
    ], dtype=np.float32)

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
    
    rmat, _ = cv2.Rodrigues(rvec)
    camera_position = np.dot(rmat, np.array([[1], [1], [2]])) + tvec
    distance = np.linalg.norm(camera_position) * square_size

    if distance > 4:
        v = 255
    else:
        v = int((distance / 4) * 255)

    h = 60 # TODO: change this so it reflects the angle the polygon has

    hsv_color = np.uint8([[[h, 255, v]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

    poly_points = [(int(point[0]), int(point[1])) for point in points[4:8]]
    # Rearrange order so polygon is square.
    poly_points[2], poly_points[3] = poly_points[3], poly_points[2]

    cv2.fillConvexPoly(
        img,
        np.asarray(poly_points),
        (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
    )

    top_middle = [int(points[8][0]), int(points[8][1])]
    cv2.circle(img, (top_middle[0], top_middle[1]), 0, (0, 0, 255))
    cv2.putText(
        img,
        f"{distance}",
        (top_middle[0], top_middle[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255)
    )


def draw_axis_cube(
    source: str | cv2.typing.MatLike,
    rvec: cv2.typing.MatLike | None, 
    tvec: cv2.typing.MatLike | None, 
    mtx: cv2.typing.MatLike, 
    dist: cv2.typing.MatLike
)-> cv2.typing.MatLike:
    """
    Draw an axis and a cube on top of an image.

    :param source: Path to an image or an actual image.
    :type source: str | cv2.typing.MatLike
    :param rvec: Rotation vector.
    :type rvec: cv2.typing.MatLike | None
    :param tvec: Translation vector.
    :type tvec: cv2.typing.MatLike | None
    :param mtx: Calibration matrix.
    :type mtx: cv2.typing.MatLike
    :param dist: Distance matrix.
    :type dist: cv2.typing.MatLike
    :return: The modified image
    :rtype: MatLike
    """
    if type(source) == str:
        img = cv2.imread(source, 1)
    else:
        img = source.copy()

    if rvec is None or tvec is None:
        return img
    draw_axis(img, rvec, tvec, mtx, dist)
    draw_cube(img, rvec, tvec, mtx, dist)
    return img

def display_axis_cube(
    source: str | cv2.typing.MatLike,
    rvec: cv2.typing.MatLike | None, 
    tvec: cv2.typing.MatLike | None, 
    mtx: cv2.typing.MatLike, 
    dist: cv2.typing.MatLike,
    win_name: str,
    wait: bool
)-> None:
    """
    Display an axis and a cube on top of an image.

    :param source: Path to an image or an actual image.
    :type source: str | cv2.typing.MatLike
    :param rvec: Rotation vector.
    :type rvec: cv2.typing.MatLike | None
    :param tvec: Translation vector.
    :type tvec: cv2.typing.MatLike | None
    :param mtx: Calibration matrix.
    :type mtx: cv2.typing.MatLike
    :param dist: Distance matrix.
    :type dist: cv2.typing.MatLike
    :param win_name: Window name.
    :type win_name: str 
    :type wait: If true, handle as stationary image, if false, 
        handle as video.
    :type wait: bool 
    :return: The modified image
    :rtype: MatLike
    """
    img = draw_axis_cube(source, rvec, tvec, mtx, dist)
    cv2.imshow(win_name, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def display_axis_cube_video(
    capture: cv2.VideoCapture,
    mtx: cv2.typing.MatLike, 
    dist: cv2.typing.MatLike,
    win_name: str,
)-> None:
    """
    Display an axis and a cube on top of a VideoCapture object..

    :param capture: OpenCV VideoCapture object
    :type capture: cv2.VideoCapture
    :param rvec: Rotation vector.
    :type rvec: cv2.typing.MatLike | None
    :param tvec: Translation vector.
    :type tvec: cv2.typing.MatLike | None
    :param mtx: Calibration matrix.
    :type mtx: cv2.typing.MatLike
    :param dist: Distance matrix.
    :type dist: cv2.typing.MatLike
    :param win_name: Window name.
    :type win_name: str 
        handle as video.
    :return: The modified image
    :rtype: MatLike
    """
    print("\nPress [esc] or [q] to close the window.")
    while True:
        _, frame = capture.read()
        display_axis_cube(
            frame, 
            *get_rvec_tvec(frame, mtx, dist),
            mtx, 
            dist, 
            win_name, 
            False
        )
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
