import cv2
import os

from calibration import calibration, extrinsics
from display_axis import draw_cube, draw_axis


# Load info of chessboard
checkerboard = cv2.FileStorage(
    "assignment_2/data/checkerboard.xml",
    cv2.FileStorage_READ
)
# Pattern size of chessboard
PATTERN_SIZE = [
    int(checkerboard.getNode("CheckerBoardWidth").real()),
    int(checkerboard.getNode("CheckerBoardHeight").real())
]
# Length of a single square of a chessboard (in meters)
SQUARE_SIZE = float(checkerboard.getNode("CheckerBoardSquareSize").real()) / \
    1000


def main()-> None:
    
    ####################################################################
    #                      Calibrate the cameras.                      #
    ####################################################################
    calibration(
        "assignment_2/data/",
        PATTERN_SIZE,
        SQUARE_SIZE
    )

    ####################################################################
    #                   Calculate camera extrinsics.                   #
    ####################################################################
    extrinsics(
        "assignment_2/data/",
        PATTERN_SIZE,
        SQUARE_SIZE
    )

    source = "assignment_2/data/"
    cameras = [
        source + folder for folder in os.listdir(source) \
            if os.path.isdir(source + folder)
    ]

    ####################################################################
    #                        Plot axis on videos                       #
    ####################################################################
    for camera in cameras:
        vid = cv2.VideoCapture(camera + "/video.avi")
        _, img = vid.read()

        camera_config = cv2.FileStorage(
            camera + "/config.xml",
            cv2.FileStorage_READ
        )
        mtx = camera_config.getNode("CameraMatrix").mat()
        rvec = camera_config.getNode("RotationVec").mat()
        tvec = camera_config.getNode("TranslationVec").mat()
        dist = camera_config.getNode("DistortionCoeffs").mat()

        draw_axis(
            img,
            rvec,
            tvec,
            mtx,
            dist,
            SQUARE_SIZE
        )
        cv2.imshow(f"cube: {camera[-4:]}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
