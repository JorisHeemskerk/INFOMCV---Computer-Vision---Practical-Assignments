import cv2
import numpy as np


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

def fit_gaussians(
    stacked_video: cv2.typing.MatLike
)-> tuple[np.ndarray, np.ndarray]:
    """
    Calculate both the mean and variances for each pixel in the image,
    based on the entire video.

    :param stacked_video: The entire video, stacked by frames on axis 0.
    :type stacked_video: cv2.typing.MatLike
    :return: Numpy array with all means and another with all variances.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    return np.mean(stacked_video, axis=0), np.var(stacked_video, axis=0)

def create_foreground_mask(
    stacked_video: cv2.typing.MatLike, 
    means: np.ndarray, 
    variances: np.ndarray,
    thresholds: tuple[float, float, float],
    minimum_std: float=5.0
)-> np.ndarray:
    """
    Apply the fitted Gaussians to the foreground to create a mask.

    For each frame in the stacked foreground video, compare each pixel
    with the mean. If this value is greater than `k` times the standard-
    deviation, this pixel is considered changed. `k` here stands for the
    specific threshold for the given colour channel. If one of the
    pixels in one of the colour channels is considered foreground, this
    will be seen as true for the entire image. In the output, each `on` 
    pixel is considered foreground and each `off` pixel is background.
    
    :param stacked_video: The entire video, stacked by frames on axis 0.
        NOTE: Feed the video in HSV format.
    :type stacked_video: cv2.typing.MatLike
    :param means: Means for the fitted Gausians, same shape as 1 frame.
    :type means: np.ndarray
    :param variances: Variances for the fitted Gausians, same shape as 1
        frame.
    :type variances: np.ndarray
    :param thresholds: List of 3 threshold values for Hue, Value & 
        Saturation, respectively.
    :type thresholds: tuple[float, float, float]
    :param minimum_std: When the standard deviation becomes too small,
        it results in tiny changes being considered as foreground. This
        minimum combats this behavior. (DEFAULT=5.0)
    :type minimum_std: float
    :return: A stack of masks per frame on axis 0.
    :rtype: np.ndarray
    """
    assert stacked_video.shape[1] == means.shape[0] == variances.shape[0], \
        "Heights of video, means and variances do not match"
    assert stacked_video.shape[2] == means.shape[1] == variances.shape[1], \
        "Widths of video, means and variances do not match"
    
    # Convert to float32 to prevent uint8 overflow...
    stacked_video = stacked_video.astype(np.float32)
    means = means.astype(np.float32)

    difference = np.abs(stacked_video - means)
    stds = np.sqrt(variances)
    # Set the minimum STD to `minimum_std`, to prevent very very minor 
    # fluctuations from being seen as foreground.
    stds = np.maximum(stds, minimum_std) 
    foreground_channels = difference > stds * np.array(thresholds)

    # If one of the channels detects a foreground, all of them do.
    return np.any(foreground_channels, axis=3)

def foreground_mask_to_video(
    destination: str, 
    foreground_mask: np.ndarray
)-> None:
    """
    Writes foreground mask to .avi video. 
    
    This function only works when `foreground_mask` contains 0's and 
    1's, while the dtype does not matter.

    :param destination: The output directory/path.
    :type destination: str
    :param foreground_mask: A foreground masked image.
    :type foreground_mask: np.ndarray
    """
    foreground_grey = (foreground_mask.astype(np.uint8)) * 255
    frames, height, width = foreground_grey.shape
    outfile = cv2.VideoWriter(
        destination,
        fourcc=cv2.VideoWriter.fourcc(*'DIVX'),
        fps=50, # Same as input.
        frameSize=[width, height],
        isColor=False
    )
    for i in range(frames):
        outfile.write(foreground_grey[i])
    outfile.release()


if __name__ == "__main__": # TODO: Move to main.py or something, idk.
    stacked_background_video = stack_video_frames(cv2.VideoCapture("assignment_2/data/cam1/background.avi"))
    stacked_foreground_video = stack_video_frames(cv2.VideoCapture("assignment_2/data/cam1/video.avi"))
    print(stacked_background_video.shape)
    means, variances = fit_gaussians(stacked_background_video)
    mask = create_foreground_mask(stacked_foreground_video, means, variances, thresholds=[2, 3, 4])
    foreground_mask_to_video("assignment_2/data/cam1/foreground_mask.avi", mask)
