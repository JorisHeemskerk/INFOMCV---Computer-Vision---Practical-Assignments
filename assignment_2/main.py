import cv2

from calibration import calibration


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
        1,
        PATTERN_SIZE,
        SQUARE_SIZE
    )


if __name__ == "__main__":
    main()
