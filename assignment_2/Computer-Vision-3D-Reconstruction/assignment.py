import glm
import numpy as np
import cv2

import sys, os
# Make python able to access files from other directories.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from background_subtraction import Thresholds, create_foreground_mask
from dataclasses import dataclass


block_size = 1.0
VOXEL_SIZE = 3 / 128 # width = 3m / 128 squares
ORIGIN = np.array([-1.2, -1.4, -1.5]) # in meters


@dataclass
class CameraConfig:
    camera: str
    mtx: cv2.typing.MatLike
    rvec: cv2.typing.MatLike
    tvec: cv2.typing.MatLike
    dist: cv2.typing.MatLike
    thresholds: Thresholds
    means: np.ndarray
    variances: np.ndarray

def get_camera_configs()-> list[CameraConfig]:
    """
    Extract the camera configurations from the different storage
    locations and save it to a list of CameraConfig objects.

    :returns: List of 4 CameraConfig objects.
    :rtype: list[CameraConfig]
    """
    all_thresholds = [
        Thresholds(
            h_top=np.float64(5.928571428571429), 
            h_bot=np.float64(10.0), 
            s_top=np.float64(8.642857142857142), 
            s_bot=np.float64(7.2857142857142865), 
            v_top=np.float64(7.2857142857142865), 
            v_bot=np.float64(10.0)
        ),
        Thresholds(
            h_top=np.float64(10.0), 
            h_bot=np.float64(10.0), 
            s_top=np.float64(10.0), 
            s_bot=np.float64(4.571428571428571), 
            v_top=np.float64(10.0), 
            v_bot=np.float64(4.571428571428571)
        ),
        Thresholds(
            h_top=np.float64(10.0), 
            h_bot=np.float64(10.0), 
            s_top=np.float64(10.0), 
            s_bot=np.float64(10.0), 
            v_top=np.float64(10.0), 
            v_bot=np.float64(10.0)
        ),
        Thresholds(
            h_top=np.float64(8.642857142857142), 
            h_bot=np.float64(8.642857142857142), 
            s_top=np.float64(8.642857142857142), 
            s_bot=np.float64(8.642857142857142), 
            v_top=np.float64(5.928571428571429), 
            v_bot=np.float64(10.0)
        )
    ]
    camera_configs: list[CameraConfig] = []
    for i, camera in enumerate(["cam1", "cam2", "cam3", "cam4"]):
        config = cv2.FileStorage(
            f"../data/{camera}/config.xml",
            cv2.FileStorage_READ
        )
        camera_configs.append(CameraConfig(
            camera=camera,
            mtx=config.getNode("CameraMatrix").mat(),
            rvec=config.getNode("RotationVec").mat(),
            tvec=config.getNode("TranslationVec").mat(),
            dist=config.getNode("DistortionCoeffs").mat(),
            thresholds=all_thresholds[i],
            means=np.load(f"../data/{camera}/means.npy"),
            variances=np.load(f"../data/{camera}/variances.npy")
        ))
    return camera_configs

CAMERA_CONFIGS = get_camera_configs()

def get_masked_frame_from_all_cameras(
    frame_id: int,
    camera_configs: list[CameraConfig] | None=None
)-> list[cv2.typing.MatLike]:
    """
    Extracts a masked foreground-background view from each camera at
    the provided frame ID.

    :param frame_id: The index of the desired frame.
    :type frame_id: int
    :param camera_configs: List of camera configurations.
    :type camera_configs: list[CameraConfig] | None
    :returns: List of frames with applied masks.
    :rtype: list[MatLike]
    """
    masked_frames: list[cv2.typing.MatLike] = []
    for config in \
        camera_configs if camera_configs is not None else CAMERA_CONFIGS:
        source = cv2.VideoCapture(f"../data/{config.camera}/video.avi")
        video_length = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_id < video_length, \
            f"\033[31mRequesting a frame ({frame_id}) that is not present in" \
            f" camera {config.camera}...\033[37m"
        # Advance the video until the desired frame.
        source.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, img = source.read()
        if not success:
            raise RuntimeError(f"\033[31mFrame {frame_id} could not be read.")
        mask = create_foreground_mask(
            # Function expects multiple frames for parallel processing,
            # but since each frame has a different config, we cant.
            np.array([img]), 
            config.means, 
            config.variances, 
            config.thresholds
        )
        masked_frames.append(mask)
    return masked_frames

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

def set_voxel_positions(
    width: int, 
    height: int, 
    depth: int
)-> tuple[list[list[int]] | np.ndarray, list[list[int]] | np.ndarray]:
    """
    Calculates which voxels to display for a given frame.

    Does this by first extracting a foreground-background frame from 
    each camera in the config. Then, it instantiates a grid based on IRL
    coordinates, along with a grid of the voxel-world coordinates. Then,
    per voxel, the point to which the 3D coordinate projects to in the
    2D space is checked for each camera. If all cameras see this point,
    the voxel is considered to be real, and gets drawn.

    :param width: Maximal width of the voxel space.
    :type width: int
    :param height: Maximal height of the voxel space.
    :type height: int
    :param depth: Maximal depth of the voxel space.
    :type depth: int
    :returns: Two arrays, the first of which containing x, y, z coords
        per voxel, and the latter r, g, b data for the corresponding
        colours.
    :rtype: tuple[
        list[list[int]] | np.ndarray, 
        list[list[int]] | np.ndarray
    ]
    """
    frame_id = 0
    masked_frames = get_masked_frame_from_all_cameras(frame_id)

    # The grid of IRL information (shape=(n, 3)). Between each point
    # there are steps of `VOXEL_SIZE`.
    irl_grid = np.mgrid[0:width, 0:depth, 0:height].transpose(1, 2, 3, 0)
    irl_grid = irl_grid.reshape(-1, 3).astype(np.float32) * VOXEL_SIZE
    irl_grid = irl_grid + ORIGIN

    # The grid of IRL information (shape=(n, 3)). Between each point
    # there are steps of `block_size`. This array gets used to map the
    # final drawing to, so the grid at the floor does not get distorted.
    voxel_grid = np.mgrid[0:width, 0:depth, 0:height].transpose(1, 2, 3, 0)
    voxel_grid = voxel_grid.reshape(-1, 3).astype(np.float32) * block_size
    voxel_grid[:, 2] = voxel_grid[:, 2] * -1 + height 
    voxel_grid = voxel_grid - np.array(
        [width / 2, depth / 2, 0], 
        dtype=np.float32
    )
    
    voxel_frames = []
    for camera_config, masked_frame in zip(CAMERA_CONFIGS, masked_frames): 
        points, _ = cv2.projectPoints(
            irl_grid,
            camera_config.rvec,
            camera_config.tvec,
            camera_config.mtx,
            camera_config.dist
        )
        points = points.reshape(-1, 2).astype(np.int16)
        
        # Set any out of bounds points to (0,0) (this happens to always 
        # be off, if this is not the case, choose a different point).
        points[
            (points[:, 0] < 0) | (points[:, 0] >= masked_frame.shape[2]) | \
            (points[:, 1] < 0) | (points[:, 1] >= masked_frame.shape[1])
        ] = np.array([0.0, 0.0], dtype=np.int16)

        xs = points[:, 0]
        ys = points[:, 1]
        voxel_frames.append(masked_frame[:, ys, xs][0])

    # Only pixels that are visible in all cameras count as actual.
    visible_voxels = voxel_grid[np.all(voxel_frames, axis=0)]
    # Switch the z and y axes of IRL coords because voxels expect y to 
    # be height and IRL expects z. 
    visible_voxels = visible_voxels[:, [0, 2, 1]]

    return visible_voxels, [[255,0,0]] * len(visible_voxels)



def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
