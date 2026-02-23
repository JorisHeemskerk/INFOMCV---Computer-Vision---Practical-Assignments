import cv2
import numpy as np

from tqdm import tqdm


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

def optimise_thresholds(
    stacked_video: cv2.typing.MatLike, 
    means: np.ndarray, 
    variances: np.ndarray,
    threshold_search_space: tuple[float, float, int],
    minimum_std: float=5.0,
    component_pixel_size_punishment: int=20,
    stride: int=1
)-> tuple[float, float, float]:
    """
    Optimises thresholds using changes in erosion and dilation.

    The core idea is that a perfect foreground-background subtraction
    should be minimally affected by both erosion and dilation. In 
    addition, there should also be as few floating shapes as possible
    (i.e., random white pixels or random small black holes). Using
    a grid-search we can try a multitude of different thresholds
    and see how much they change under erosion, dilation & small shapes.
    This way, the optimal thresholds are the ones that minimally change 
    the mask or create small shapes. Both are weighed equally.

    :param stacked_video: An entire video, stacked by frames on axis 0.
        NOTE: Feed the video in black and white format.
    :param stacked_video: cv2.typing.MatLike, 
    :param means: Means for the fitted Gausians, same shape as 1 frame.
    :type means: np.ndarray
    :param variances: Variances for the fitted Gausians, same shape as 1
        frame.
    :param variances: np.ndarray
    :param threshold_search_space: The input for a np.linspace. It 
        equally divides the last digit number of steps between the first
        two floats.
    :param threshold_search_space: tuple[float, float, int]
    :param minimum_std: Minimum standard deviation for masking.
        (DEFAULT=5.0)
    :param minimum_std: float
    :param component_pixel_size_punishment: When a group of pixels is
        smaller than this amount, it will be marked as negative and
        negatively impact the score. (DEFAULT=20)
    :param component_pixel_size_punishment: int
    :param stride: Steps to take through the video, only looks at every
        `stride` frames. (DEFAULT=1)
    :param stride: int
    :returns: The optimally found thresholds for Hue, Value, and 
    Saturation, respectively.
    :rtype: tuple[float, float, float]
    """
    h_thresholds = np.linspace(*threshold_search_space)
    s_thresholds = np.linspace(*threshold_search_space)
    v_thresholds = np.linspace(*threshold_search_space)

    best_score = float("inf")
    optimal_thresholds: tuple[float, float, float] = (.0, .0, .0)

    for h in tqdm(h_thresholds, desc="Outer loop for hue thresholds"):
        for s in s_thresholds:
            for v in v_thresholds:
                mask = create_foreground_mask(
                    stacked_video[::stride], 
                    means, 
                    variances, 
                    thresholds=(h, s, v),
                    minimum_std=minimum_std
                )
                mask = mask.astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                total_score = 0
                for frame in mask:
                    cleaned_frame = cv2.morphologyEx(
                        frame, 
                        cv2.MORPH_OPEN, 
                        kernel
                    )
                    difference = np.abs(frame - cleaned_frame)
                    
                    # Count the number of connected white shapes of 
                    # fewer than 20 pixels.
                    _, labels = cv2.connectedComponents(frame)
                    component_sizes = np.bincount(labels.flatten())
                    n_small_components = np.sum(
                        component_sizes < component_pixel_size_punishment
                    )

                    # Total += the normalised number of small components
                    # + the normalised morphology difference.
                    total_score += \
                        (n_small_components / len(component_sizes)) + \
                        np.mean(difference)
                if total_score < best_score:
                    best_score = total_score
                    optimal_thresholds = (h, s, v)

    return optimal_thresholds


if __name__ == "__main__": # TODO: Move to main.py or something, idk.
    stacked_background_video = stack_video_frames(cv2.VideoCapture("assignment_2/data/cam1/background.avi"))
    stacked_foreground_video = stack_video_frames(cv2.VideoCapture("assignment_2/data/cam1/video.avi"))
    print(stacked_background_video.shape)
    means, variances = fit_gaussians(stacked_background_video)

    # thresholds=[2, 3, 4]
    thresholds = optimise_thresholds(stacked_foreground_video, means, variances, threshold_search_space=(5, 10.0, 16), stride=10)
    print(thresholds)
    mask = create_foreground_mask(stacked_foreground_video, means, variances, thresholds)
    foreground_mask_to_video("assignment_2/data/cam1/foreground_mask.avi", mask)
