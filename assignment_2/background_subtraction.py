import cv2
import numpy as np
import itertools
import joblib

from dataclasses import dataclass
from tqdm import tqdm

CAMERA = "cam4"


@dataclass
class Thresholds:
    h_top: float
    h_bot: float
    s_top: float
    s_bot: float
    v_top: float
    v_bot: float

ALL_THRESHOLDS = {
        "cam1" : Thresholds(
            h_top=np.float64(5.928571428571429), 
            h_bot=np.float64(10.0), 
            s_top=np.float64(8.642857142857142), 
            s_bot=np.float64(7.2857142857142865), 
            v_top=np.float64(7.2857142857142865), 
            v_bot=np.float64(10.0)
        ),
        "cam2" : Thresholds(
            h_top=np.float64(10.0), 
            h_bot=np.float64(10.0), 
            s_top=np.float64(10.0), 
            s_bot=np.float64(6), 
            v_top=np.float64(10.0), 
            v_bot=np.float64(4.571428571428571)
        ),
        "cam3" : Thresholds(
            h_top=np.float64(10), 
            h_bot=np.float64(10), 
            s_top=np.float64(8.642857142857142), 
            s_bot=np.float64(9), 
            v_top=np.float64(5.928571428571429), 
            v_bot=np.float64(10.0)
        ),
        "cam4" : Thresholds(
            h_top=np.float64(8.642857142857142), 
            h_bot=np.float64(8.642857142857142), 
            s_top=np.float64(8.642857142857142), 
            s_bot=np.float64(8.642857142857142), 
            v_top=np.float64(5.928571428571429), 
            v_bot=np.float64(10.0)
        )
    }

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
    thresholds: Thresholds,
    minimum_std: float=5.0
)-> np.ndarray:
    """
    Apply the fitted Gaussians to the foreground to create a mask.

    For each frame in the stacked foreground video, compare each pixel
    with the mean. If this value is greater/smaller than `k` times the 
    standard deviation, this pixel is considered changed. `k` here 
    stands for the specific threshold for the given colour channel. If 
    one of the pixels in one of the colour channels is considered 
    foreground, this will be seen as true for the entire image. In the 
    output, each `on` pixel is considered foreground and each `off` 
    pixel is background.
    
    :param stacked_video: The entire video, stacked by frames on axis 0.
        NOTE: Feed the video in HSV format.
    :type stacked_video: cv2.typing.MatLike
    :param means: Means for the fitted Gausians, same shape as 1 frame.
    :type means: np.ndarray
    :param variances: Variances for the fitted Gausians, same shape as 1
        frame.
    :type variances: np.ndarray
    :param thresholds: Thresholds object containing 6 values for Hue, 
        Value & Saturation, respectively, containing 2 of each 
        (top & bottom).
    :type thresholds: Thresholds
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
    
    # Convert to float32 to prevent uint8 overflow.
    stacked_video = stacked_video.astype(np.float32)
    means = means.astype(np.float32)

    stds = np.sqrt(variances)
    # Set the minimum STD to `minimum_std`, to prevent very very minor 
    # fluctuations from being seen as foreground.
    stds = np.maximum(stds, minimum_std) 

    H = stacked_video[..., 0]
    S = stacked_video[..., 1]
    V = stacked_video[..., 2]

    mean_H = means[..., 0]
    mean_S = means[..., 1]
    mean_V = means[..., 2]

    std_H = stds[..., 0]
    std_S = stds[..., 1]
    std_V = stds[..., 2]

    h_foreground = (
        (H > mean_H + thresholds.h_top * std_H) |
        (H < mean_H - thresholds.h_bot * std_H)
    )

    s_foreground = (
        (S > mean_S + thresholds.s_top * std_S) |
        (S < mean_S - thresholds.s_bot * std_S)
    )

    v_foreground = (
        (V > mean_V + thresholds.v_top * std_V) |
        (V < mean_V - thresholds.v_bot * std_V)
    )

    # If one of the channels detects a foreground, all of them do.
    return h_foreground | s_foreground | v_foreground

def foreground_mask_to_video(
    destination: str, 
    foreground_mask: np.ndarray
)-> None:
    """
    Writes foreground mask to .avi video. 

    :param destination: The output directory/path.
    :type destination: str
    :param foreground_mask: A foreground masked image.
    :type foreground_mask: np.ndarray
    """
    if foreground_mask[0,0].dtype == np.bool:
        foreground_grey = (foreground_mask.astype(np.uint8)) * 255
    else:
        foreground_grey = foreground_mask
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

def foreground_mask_to_single_frame(
    destination: str, 
    foreground_mask: np.ndarray
)-> None:
    """
    Writes the first frame of foreground mask to file. 

    :param destination: The output directory/path.
    :type destination: str
    :param foreground_mask: A foreground masked image.
    :type foreground_mask: np.ndarray
    """
    if foreground_mask[0,0].dtype == np.bool:
        foreground_grey = (foreground_mask.astype(np.uint8)) * 255
    else:
        foreground_grey = foreground_mask

    cv2.imwrite(destination, foreground_grey[0])

def optimise_thresholds(
    stacked_video: cv2.typing.MatLike, 
    means: np.ndarray, 
    variances: np.ndarray,
    threshold_search_space: tuple[float, float, int],
    minimum_std: float=5.0,
    component_pixel_size_punishment: int=20,
    stride: int=1
)-> Thresholds:
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
    :param stacked_video: cv2.typing.MatLike
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
    :returns: The optimally found top and bottom thresholds for Hue, 
        Value, and Saturation, respectively.
    :rtype: Thresholds
    """
    def validate_thresholds(thresholds: Thresholds)-> float:
        mask = create_foreground_mask(
            stacked_video[::stride], 
            means, 
            variances, 
            thresholds=thresholds,
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
        return total_score 

    threshold_space = np.linspace(*threshold_search_space)
    possible_thresholds = itertools.product(threshold_space, repeat=6)
    print(
        f"There will be {len(threshold_space) ** 6:,} "
        "combinations validated..."
    )

    scores = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(validate_thresholds)(Thresholds(*thresholds)) 
        for thresholds in tqdm(
            possible_thresholds, 
            total=len(threshold_space) ** 6
        )
    )
    scores = np.array(scores)
    # Cache all the scores.
    np.save(f"scores_{CAMERA}.npy", scores)
    
    # Create all combinations again, because the yield object is empty.
    possible_thresholds = itertools.product(threshold_space, repeat=6)
    best_thresholds = next(
        itertools.islice(possible_thresholds, np.argmin(scores), None)
    )
    best_thresholds = Thresholds(*best_thresholds)
    return best_thresholds

def apply_post_processing(
    stacked_video: cv2.typing.MatLike,
)-> cv2.typing.MatLike:
    """
    Apply post-processing to already masked video.

    Applies erosion -> dilation, erosion -> dilation and blob detection.

    :param stacked_video: An entire video, stacked by frames on axis 0.
    :param stacked_video: cv2.typing.MatLike 
    """
    grey_video = (stacked_video.astype(np.uint8)) * 255
    kernel = np.ones((3,3), np.uint8)
    clean_video = []
    for frame in grey_video:
        frame = cv2.morphologyEx(
            frame, 
            cv2.MORPH_OPEN, 
            kernel
        )
        frame = cv2.morphologyEx(
            frame, 
            cv2.MORPH_CLOSE,
            kernel
        )

        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            frame, connectivity=8
        )
        # Zero is always the background, so we start from 1.
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        largest_component = np.zeros_like(frame)
        largest_component[labels == largest_label] = 255
                    
        if CAMERA == "cam2":
            y_min = 330
            y_max = 280

            roi = largest_component[y_max:y_min + 1, :]

            roi_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            roi_filled = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, roi_kernel)

            largest_component[y_max:y_min + 1, :] = roi_filled
        elif CAMERA == "cam3":
            y_min = 350
            y_max = 300

            roi = largest_component[y_max:y_min + 1, :]

            roi_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            roi_filled = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, roi_kernel)

            largest_component[y_max:y_min + 1, :] = roi_filled

        elif CAMERA == "cam4":
            y_min = 280
            y_max = 250

            roi = largest_component[y_max:y_min + 1, :]
            roi_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            roi_filled = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, roi_kernel)

            largest_component[y_max:y_min + 1, :] = roi_filled

        clean_video.append(largest_component)
    return np.array(clean_video)



if __name__ == "__main__": # TODO: Move to main.py or something, idk.
    stacked_background_video = stack_video_frames(cv2.VideoCapture(f"assignment_2/data/{CAMERA}/background.avi"))
    stacked_foreground_video = stack_video_frames(cv2.VideoCapture(f"assignment_2/data/{CAMERA}/video.avi"))
    print(stacked_background_video.shape)
    means, variances = fit_gaussians(stacked_background_video)
    np.save(f"assignment_2/data/{CAMERA}/means.npy", means)
    np.save(f"assignment_2/data/{CAMERA}/variances.npy", variances)

    thresholds = ALL_THRESHOLDS[CAMERA]
    # thresholds = optimise_thresholds(stacked_foreground_video, means, variances, threshold_search_space=(0.5, 10.0, 8), stride=20)
    mask = create_foreground_mask(stacked_foreground_video, means, variances, thresholds)
    mask = apply_post_processing(mask)
    foreground_mask_to_single_frame(f"assignment_2/data/{CAMERA}/foreground_mask.jpg", mask)
    foreground_mask_to_video(f"assignment_2/data/{CAMERA}/foreground_mask.avi", mask)
