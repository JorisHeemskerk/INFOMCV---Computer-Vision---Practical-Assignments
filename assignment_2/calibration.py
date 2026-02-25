import cv2
import os
import numpy as np

from detect_corners import \
    detect_corners, \
    automatic_corner_detector, \
    manual_corner_selector
from display_axis import draw_axis


def calibration(
    source: str,
    frames_num: int,
    pattern_size: cv2.typing.Size,
    square_size: float
)-> None:
    """
    Calibrate all cameras based on a set of corners from chessboards.
    For all camera calibrations: save or overwrite the `intrinsics.xml`
    file in the corresponding folder.

    Part of the xml saving code was inspired by:
    https://docs.opencv.org/4.x/d4/da4/group__core__xml.html#xml_storage

    :param source: Folder containing 'camX' folders where 'X' is
        replaced by the id of the camera. These folders contain an
        intrinsics.avi file.
    :type source: str
    :param frames_num: The amount of frames used for the calibration of
        each camera.
    :type frames_num: int
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :param square_size: The length of a single chessboard square.
    :type square_size: float
    """
    cameras = [
        source + folder + "/intrinsics.avi" for folder in os.listdir(source) \
            if os.path.isdir(source + folder)
    ]

    for camera in cameras:
        # Get all frames in video
        frames = stack_video_frames(camera, cv2.COLOR_BGR2GRAY)
        # Detect chessboard corners in frames
        # all_corners, img_shape = detect_corners([frames[0]], pattern_size)
        stride = np.array(range(0, frames.shape[0], 50))
        all_corners, img_shape = detect_corners(frames[stride], pattern_size)
        
        # Calibrate the camera
        _, mtx, dist, _, _ = calibrate_camera(
            all_corners, 
            img_shape, 
            pattern_size, 
            square_size
        )

        xml = cv2.FileStorage(
            camera.replace(".avi", ".xml"),
            cv2.FILE_STORAGE_WRITE
        )
        # Convert to float32 as per template
        mtx = mtx.astype(np.float32)
        dist = dist.astype(np.float32)
        
        xml.write("CameraMatrix", mtx)
        xml.write("DistortionCoeffs", dist)

        xml.release()


def stack_video_frames(
    source: str | cv2.VideoCapture,
    colour: int | None=None
)-> cv2.typing.MatLike:
    """
    Read video and stack into frames.

    Read video frame by frame and load them all into a numpy array of 
    shape `(vid_length, vid_height, vid_width, x)`, where the final
    x stands for the colour channel. When this is grey, the x is gone.

    :param source: Path to input video or an actual input video
    :type source: str | cv2.typing.MatLike
    :param colour: Cv2 colour conversion code or None if original colour
        is needed.
    :type colour: int | None
    :return: The stacked video as numpy array
    :rtype: cv2.typing.MatLike
    """
    if type(source) == str:
        source = cv2.VideoCapture(source)

    stacked_video = []
    video_length = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(video_length):
        success, img = source.read()
        if not success:
            print(
                f"\033[31mFrame {i + 1} / {video_length} could not be read, "
                "and will therefore be skipped.\033[37m"
            )
        else:
            if colour:
                stacked_video.append(cv2.cvtColor(img, colour))
            else:
                stacked_video.append(img)
    return np.array(stacked_video)


def calibrate_camera(
    all_corners: list[list[tuple[float, float]]],
    img_shape: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size,
    square_size: float
)-> tuple[
    float,
    cv2.typing.MatLike,
    cv2.typing.MatLike,
    tuple[cv2.typing.MatLike],
    tuple[cv2.typing.MatLike]
]:
    """
    Calibrate a camera based on a set of detected corners from chess 
    boards.

    First, this function establishes real-world xyz coordinates. These
    are used together with the detected `all_corners`, in order to
    estimate the camera properties.

    :param all_corners: Container for all the corners of all the images.
    :type all_corners: list[list[tuple[float, float]]]
    :param img_shape: Shape (x,y) of the original images in pixels.
    :type img_shape: cv2.typing.MatLike
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :param square_size: The length of a single chessboard square.
    :type square_size: float
    :return: Ret, mtx, dist, rvecs, tvecs.
    :rtype: tuple[float, MatLike, MatLike, tuple[MatLike], tuple[MatLike]]
    """
    # Create real-world coordinates, starting at 0,0,0 for the top-left
    # point, taking steps of 1 for each new intersection.
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size
    real_points = [objp.copy() for _ in range(len(all_corners))]

    return cv2.calibrateCamera(
        real_points,
        all_corners,
        img_shape[::-1],
        None,
        None
    )


def get_rvec_tvec(
    source: str | cv2.typing.MatLike,
    mtx: cv2.typing.MatLike,
    dist: cv2.typing.MatLike,
    pattern_size: cv2.typing.Size,
    square_size: float
)-> tuple[cv2.typing.MatLike | None, cv2.typing.MatLike | None]:
    """
    Get the rotation and translation vectors form an image.

    Calculate the rotation and translation vectors from an image using
    the calibrated matrix and distances. This function detects the
    chessboard corners in an image and calculates the vectors. If the
    provided image comes from a file, the user may be prompted to 
    manually detect the corners. If an already loaded image is passed,
    and no corners can automatically be detected, the function will
    return two None vectors.

    :param source: Path to input image or an actual input image
    :type source: str | cv2.typing.MatLike
    :param mtx: The camera intrinsics matrix
    :type mtx: cv2.typing.MatLike
    :param dist: The distortion coefficients
    :type dist: cv2.typing.MatLike
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :param square_size: The length of a single chessboard square.
    :type square_size: float
    :return: The rotation vector and translation vector of the input 
        image
    :rtype: tuple[MatLike, MatLike]
    """
    if type(source) == str:
        img = cv2.imread(source, 1)
        corners, _ = detect_corners(source, None, pattern_size)
        # Extract the first and only corners element.
        corners = corners[0]
    else:
        img = source.copy()
        success, corners, _ = automatic_corner_detector(img, pattern_size)
        # if success == 0:
        #     print("Couldn't find corners")
        #     return None, None
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

    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size

    _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    return rvec, tvec


def extrinsics(
    source: str,
    pattern_size: cv2.typing.Size,
    square_size: float
)-> None:
    """
    Calculate the extrinsics of all cameras.

    For all cameras: save the camera extrinsics and intrinsics or
    overwrite the `config.xml` file in the corresponding folder.

    :param source: Folder containing 'camX' folders where 'X' is
        replaced by the id of the camera. These folders contain an
        intrinsics.avi file.
    :type source: str
    :param pattern_size: The size of the chessboard (n_rows x n_columns)
        counted as the number of inner corners.
    :type pattern_size: Size
    :param square_size: The length of a single chessboard square.
    :type square_size: float
    """
    cameras = [
        source + folder for folder in os.listdir(source) \
            if os.path.isdir(source + folder)
    ]

    for camera in cameras:
        # Get all frames in video
        frames = stack_video_frames(
            camera + "/checkerboard.avi",
            None
        )

        # Get camera intrinsics
        intrinsics = cv2.FileStorage(
            camera + "/intrinsics.xml",
            cv2.FileStorage_READ
        )
        mtx = intrinsics.getNode("CameraMatrix").mat()
        dist = intrinsics.getNode("DistortionCoeffs").mat()

        rvec, tvec = get_rvec_tvec(
            frames[0],
            mtx,
            dist,
            pattern_size,
            square_size
        )
        rvec = rvec.astype(np.float32)
        tvec = tvec.astype(np.float32)

        draw_axis(
            frames[0],
            rvec,
            tvec,
            mtx,
            dist,
            square_size
        )

        cv2.imshow("window", frames[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        xml = cv2.FileStorage(
            camera + "/config.xml",
            cv2.FILE_STORAGE_WRITE
        )
        
        xml.write("CameraMatrix", mtx)
        xml.write("DistortionCoeffs", dist)
        xml.write("RotationVec", rvec)
        xml.write("TranslationVec", tvec)

        xml.release()
        
        exit()
