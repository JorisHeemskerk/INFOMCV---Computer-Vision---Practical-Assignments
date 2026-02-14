import cv2
import os
import re
import datetime
import numpy as np

from autmatic_corners import automatic_corner_detector
from manual_corners import manual_corner_selector


def detect_corners(
    folder: str,
    output_folder: str
)-> tuple[list, cv2.typing.MatLike]:
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
    
    :param folder: Folder in which input images are saved.
    :type folder: str
    :param output_folder: Folder to which output is saved. The corners
    will be saved in a subdirectory called "corners". And the corner
    points will also be saved in a subdirectory called "data"
    :type output_folder: str
    :return: List containing all the found corners of all images in 
        folder.
    :rtype: list
    """
    filenames = sorted([
        filename for filename in os.listdir(folder) if filename[0] != "."
    ], key=lambda s: int(re.search(r'\d+', s).group()))
    os.mkdir(output_folder + "corners/")

    corners_all_images = []
    for filename in filenames:
        success, corners, img = automatic_corner_detector(folder + filename)
        while success == 0:
            print(
                f"Corners were not automatically or fully manually detected in"
                f" image {filename}.\nPlease manually click on the four "
                "corners and then close the image."
            )
            success, corners, img = manual_corner_selector(folder + filename)
        # Correct corners by looking at surrounding pixels.
        corners_corrected = cv2.cornerSubPix(
            cv2.cvtColor(cv2.imread(folder + filename, 1), cv2.COLOR_BGR2GRAY),
            corners,
            (11,11),
            (-1,-1), # No dead region.
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    
        corners_all_images.append(corners_corrected)
        cv2.imwrite(output_folder + "corners/" + filename, img)

    os.mkdir(output_folder + "data/")
    np.save(
        output_folder + "data/corners_all_images.npy", 
        np.array(corners_all_images)
    )
    return corners_all_images, img.shape[:2]


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
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    real_points = [objp.copy() for _ in range(len(all_corners))]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        real_points,
        all_corners,
        img_shape[::-1],
        None,
        None
    )
    return ret, mtx, dist, rvecs, tvecs


def process_image(
    img_path: str,
    ret: float,
    mtx: cv2.typing.MatLike,
    dist: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size=[9,6]
)-> None:
    """
    TODO
    """
    success, corners, img = automatic_corner_detector(img_path)

    if success == 0:
        return # TODO 

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


def main()-> None:
    # Create output folder for the finished images.
    output_folder = "assignment_1/output/run_" + \
        f"{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}/"
    os.mkdir(output_folder)

    # all_corners, img_shape = detect_corners("assignment_1/data/all/", output_folder)
    all_corners, img_shape = detect_corners("assignment_1/data/auto/", output_folder)
    # all_corners, img_shape = detect_corners("assignment_1/data/mix/", output_folder)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(all_corners, img_shape)

    process_image(
        "assignment_1/data/test/img_1.jpg",
        ret,
        mtx,
        dist,
    )


if __name__ == "__main__":
    main()
