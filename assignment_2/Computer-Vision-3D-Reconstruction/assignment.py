import glm
import numpy as np
import cv2
import sys, os
# Make python able to access files from other directories.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from background_subtraction import \
    Thresholds, \
    create_foreground_mask, \
    stack_video_frames, \
    apply_post_processing, \
    ALL_THRESHOLDS
from dataclasses import dataclass
from engine.config import config


block_size = 1.0
VOXEL_SIZE = 1.5 / 128 # height = 1.5m / 128 voxels
# ORIGIN = np.array([-1.2, -1.4, -1.5]) # in meters
ORIGIN = np.array([-.15, -.4, -1.5]) # in meters


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
    camera_configs: list[CameraConfig] = []
    for camera in ["cam1", "cam2", "cam3", "cam4"]:
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
            thresholds=ALL_THRESHOLDS[camera],
            means=np.load(f"../data/{camera}/means.npy"),
            variances=np.load(f"../data/{camera}/variances.npy")
        ))
    return camera_configs

########################################################################
# NOTE: Constant defined here, due to dependency on above function(s). #
########################################################################
CAMERA_CONFIGS = get_camera_configs()

def get_all_masked_frames_from_all_cameras(
    camera_configs: list[CameraConfig]=CAMERA_CONFIGS
)-> np.ndarray:
    """
    Extracts a masked foreground-background view from each camera at
    the provided frame ID.

    :param camera_configs: List of camera configurations. 
        (DEFAULT=CAMERA_CONFIGS)
    :type camera_configs: list[CameraConfig] 
    :returns: List of videos with applied masks.
    :rtype: np.ndarray
    """
    print("\033[32mAttempting to pre-load and pre-mask all videos...\033[37m")
    masked_videos = []
    for config in camera_configs:
        video = stack_video_frames(f"../data/{config.camera}/video.avi")
        mask = create_foreground_mask(
            video, 
            config.means, 
            config.variances, 
            config.thresholds
        )
        mask = apply_post_processing(mask, config.camera)
        masked_videos.append(mask)
    return np.array(masked_videos)

def get_all_coloured_frames_from_all_cameras(
    camera_configs: list[CameraConfig]=CAMERA_CONFIGS
)-> np.ndarray:
    """
    Extracts all coloured videos from all camera perspectives.

    Converts the format to RGB, then normalises the pixel values to be 
    between 0 and 1, as required per the environment.

    :param camera_configs: List of camera configurations. 
        (DEFAULT=CAMERA_CONFIGS)
    :type camera_configs: list[CameraConfig] 
    :returns: List of videos.
    :rtype: np.ndrarray
    """
    print("\033[32mAttempting to pre-load all coloured videos...\033[37m")
    all_videos: list[cv2.typing.MatLike] = []
    for config in camera_configs:
        video = stack_video_frames(
            f"../data/{config.camera}/video.avi", 
            cv2.COLOR_BGR2RGB
        ).astype(np.float32)
        # Normalise all pixel values between 0 and 1.
        video /= 255 
        all_videos.append(video)
    return np.array(all_videos)

########################################################################
# NOTE: Constant defined here, due to dependency on above function(s). #
########################################################################
try:
    print("\033[32mAttempting load (masked) videos from file...\033[37m")
    CACHED_MASKED_VIDEOS = np.load("cached_masked_videos.npy")
    CACHED_ORIGINAL_VIDEOS = np.load("cached_original_videos.npy")
except:
    print("\033[31mFailed!\033[37m")
    CACHED_MASKED_VIDEOS = get_all_masked_frames_from_all_cameras()
    CACHED_ORIGINAL_VIDEOS = get_all_coloured_frames_from_all_cameras()
    np.save("cached_masked_videos.npy", CACHED_MASKED_VIDEOS)
    np.save("cached_original_videos.npy", CACHED_ORIGINAL_VIDEOS)

def get_masked_frame_from_all_cameras(
    frame_id: int,
    camera_configs: list[CameraConfig]=CAMERA_CONFIGS,
    cache: np.ndarray | None=CACHED_MASKED_VIDEOS
)-> np.ndarray:
    """
    Extracts a masked foreground-background view from each camera at
    the provided frame ID. Uses Cached frames if possible.

    :param frame_id: The index of the desired frame.
    :type frame_id: int
    :param camera_configs: List of camera configurations.
        (DEFAULT=CAMERA_CONFIGS)
    :type camera_configs: list[CameraConfig]
    :param cache: Cache containing pre-processed and masked videos.
    :type cache: np.ndarray | None
    :returns: List of frames with applied masks.
    :rtype: np.ndarray
    """
    if cache is not None:
        return cache[:, frame_id]
    masked_frames: list[cv2.typing.MatLike] = []
    for config in camera_configs:
        source = cv2.VideoCapture(f"../data/{config.camera}/video.avi")
        video_length = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_id < video_length, \
            f"\033[31mRequesting a frame ({frame_id}) that is not present in" \
            f" camera {config.camera}...\033[37m"
        # Advance the video until the desired frame.
        source.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, img = source.read()
        if not success:
            raise RuntimeError(
                f"\033[31mFrame {frame_id} could not be read.\033[37m"
            )
        mask = create_foreground_mask(
            # Function expects multiple frames for parallel processing,
            # but since each frame has a different config, we cant.
            np.array([img]), 
            config.means, 
            config.variances, 
            config.thresholds
        )[0]
        masked_frames.append(mask)
    return np.array(masked_frames)

def generate_irl_and_voxel_grid(
    width: int, 
    height:int,
    depth: int 
)-> tuple[np.ndarray, np.ndarray]:
    """
    Generate two grids, one for IRL information (shape=(n, 3)). Between 
    each point in this grid, there are steps of `VOXEL_SIZE`. The origin
    is the origin defined in `origin`. The second, or IRL grid contains
    the IRL information (shape=(n, 3)). Between each point in this grid
    there are steps of `block_size`. This array gets used to map the
    final drawing to, so the grid at the floor does not get distorted.

    :param width: Maximal width of the voxel space.
    :type width: int
    :param height: Maximal height of the voxel space.
    :type height: int
    :param depth: Maximal depth of the voxel space.
    :type depth: int
    """
    irl_grid = np.mgrid[0:width, 0:depth, 0:height].transpose(1, 2, 3, 0)
    irl_grid = irl_grid.reshape(-1, 3).astype(np.float32) * VOXEL_SIZE
    irl_grid = irl_grid + ORIGIN

    voxel_grid = np.mgrid[0:width, 0:depth, 0:height].transpose(1, 2, 3, 0)
    voxel_grid = voxel_grid.reshape(-1, 3).astype(np.float32) * block_size
    voxel_grid[:, 2] = voxel_grid[:, 2] * -1 + height 
    voxel_grid = voxel_grid - np.array(
        [width / 2, depth / 2, 0], 
        dtype=np.float32
    )
    # Switch the z and y axes of voxel coords because voxels expect y to 
    # be height and IRL expects z. 
    voxel_grid = voxel_grid[:, [0, 2, 1]]

    return irl_grid, voxel_grid

########################################################################
# NOTE: Constant defined here, due to dependency on above function(s). #
########################################################################
IRL_GRID, VOXEL_GRID = generate_irl_and_voxel_grid(
    config['world_width'], 
    config['world_height'], 
    config['world_width']
)

def pre_project_points(
    irl_grid: np.ndarray, 
    camera_configs: list[CameraConfig]=CAMERA_CONFIGS,
    image_dims: tuple[int, int]=(486, 644)
)-> tuple[np.ndarray, np.ndarray]:
    """
    Project all points in `irl_grid` onto an image, and return the 
    corresponding pixel coordinates.

    :param irl_grid: The grid with IRL coordinates.
    :type irl_grid: np.ndarray
    :param camera_configs: List of camera configurations.
        (DEFAULT=CAMERA_CONFIGS)
    :type camera_configs: list[CameraConfig]    
    :param image_dims: Image dimensions. (DEFAULT=(486, 644))
    :type image_dims: tuple[int, int]
    :returns: The x coordinates per camera, and the ys too.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    xss = []
    yss = []
    for camera_config in tqdm(camera_configs, desc="Pre-projecting points." ): 
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
            (points[:, 0] < 0) | (points[:, 0] >= image_dims[1]) | \
            (points[:, 1] < 0) | (points[:, 1] >= image_dims[0])
        ] = np.array([0.0, 0.0], dtype=np.int16)

        xs = points[:, 0]
        ys = points[:, 1]
        xss.append(xs)
        yss.append(ys)
    return np.array(xss), np.array(yss) 

########################################################################
# NOTE: Constant defined here, due to dependency on above function(s). #
########################################################################
YSS, XSS = pre_project_points(IRL_GRID, CAMERA_CONFIGS)
CAMERA_INDEX_MASK = np.arange(len(CAMERA_CONFIGS))[:, None]
    
def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
from dataclasses import dataclass, field

EXPLOSION_ORIGIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)

@dataclass
class ExplosionState:
    """
    Holds the per-voxel velocity state across frames.
    Velocities are keyed by a rounded coordinate tuple so they can be 
    matched across calls even if voxel arrays are reordered.
    """
    velocities: dict = field(default_factory=dict)

    # Physics constants
    blast_strength:  float = 10.2   # Initial outward impulse magnitude
    gravity:         float = 0.05  # Downward acceleration per tick (Y axis)
    damping:         float = 0.97  # Velocity decay factor per tick (air drag)
    floor_y:         float = -64.0 # Y below which voxels bounce
    bounce_restitution: float = 0.35 # Energy retained on floor bounce

    def _coord_key(self, pos: np.ndarray) -> tuple:
        """Round position to a stable hashable key."""
        return tuple(np.round(pos, 2))

    def _init_velocity(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute the initial blast velocity for a freshly-encountered voxel.

        The direction is the unit vector from the explosion origin to the 
        voxel. A small random jitter breaks symmetry so the result looks 
        organic rather than perfectly radial.
        """
        direction = pos - EXPLOSION_ORIGIN
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            # Voxel sits exactly at the origin — send it straight up.
            direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / dist

        # Scale impulse: closer voxels get a stronger kick.
        falloff = 1.0 / (1.0 + 0.05 * dist)
        speed   = self.blast_strength * falloff

        jitter  = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        return (direction + jitter) * speed

    def step(
        self,
        positions: np.ndarray,   # shape (N, 3)
        colours:   np.ndarray,   # shape (N, 3)  — passed through unchanged
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply one physics tick to every voxel.

        For each voxel:
          1. Look up (or initialise) its velocity in the state dictionary.
          2. Apply gravity.
          3. Integrate: new_pos = pos + velocity.
          4. Bounce off the floor.
          5. Apply damping.
          6. Store the updated velocity.

        :param positions: Voxel XYZ coordinates, shape (N, 3).
        :param colours:   Corresponding RGB values,  shape (N, 3).
        :returns: (new_positions, colours) with the same shapes.
        """
        new_positions = positions.copy().astype(np.float32)

        for i, pos in enumerate(positions):
            key = self._coord_key(pos)

            # First time we see this voxel: assign blast velocity.
            if key not in self.velocities:
                self.velocities[key] = self._init_velocity(pos)

            vel = self.velocities[key].copy()

            # --- Physics update ---
            vel[1] -= self.gravity          # gravity pulls down Y

            new_pos = pos + vel

            # Floor bounce: reflect Y velocity with energy loss.
            if new_pos[1] < self.floor_y:
                new_pos[1]  = self.floor_y
                vel[1]      = abs(vel[1]) * self.bounce_restitution

            vel *= self.damping             # air resistance

            # Persist updated velocity under the *new* key so subsequent
            # calls continue tracking the moved voxel.
            new_key = self._coord_key(new_pos)
            self.velocities[new_key] = vel

            # Clean up the old key if the voxel actually moved.
            if new_key != key:
                self.velocities.pop(key, None)

            new_positions[i] = new_pos

        return new_positions, colours


# ---------------------------------------------------------------------------
# Module-level singleton — create once, call repeatedly each frame.
# ---------------------------------------------------------------------------
_explosion_state = ExplosionState()

def explode_voxels(
    positions: np.ndarray,
    colours:   np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Public entry point.  Drop-in replacement for the positions returned 
    by `set_voxel_positions`:

        positions, colours = set_voxel_positions(frame_id)
        positions, colours = explode_voxels(positions, colours)

    Call once per rendered frame.  The internal state accumulates across
    calls so the explosion evolves over time automatically.

    :param positions: Voxel XYZ coordinates from `set_voxel_positions`.
    :param colours:   Matching RGB colours   from `set_voxel_positions`.
    :returns: (new_positions, colours)
    """
    return _explosion_state.step(positions, colours)


def reset_explosion() -> None:
    """Clear all velocities so the explosion can be re-triggered."""
    global _explosion_state
    _explosion_state = ExplosionState()

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

def set_voxel_positions(
    frame_id: int,
    do_colour: bool=True
)-> tuple[list[list[int]] | np.ndarray, list[list[int]] | np.ndarray]:
    """
    Calculates which voxels to display for a given frame.

    Does this by first extracting a foreground-background frame from 
    each camera in the config. Then, it instantiates a grid based on IRL
    coordinates, along with a grid of the voxel-world coordinates. Then,
    per voxel, the point to which the 3D coordinate projects to in the
    2D space is checked for each camera. If all cameras see this point,
    the voxel is considered to be real, and gets drawn.
    NOTE: This description is accurate to the process, but the steps
    that can be cached are cached (all except checking if the camera 
    sees a certain point or not.).

    When `do_colour` == True, each voxel gets a colour corresponding
    to the mean of the colour in the original image space.

    :param frame_id: ID of frame to parse.
    :type frame_id: int
    :param do_colour: If True, apply colouring algorithm. (DEFAULT=TRUE)
    :type do_colour: bool
    :returns: Two arrays, the first of which containing x, y, z coords
        per voxel, and the latter r, g, b data for the corresponding
        colours.
    :rtype: tuple[
        list[list[int]] | np.ndarray, 
        list[list[int]] | np.ndarray
    ]
    """
    masked_frames = get_masked_frame_from_all_cameras(frame_id)

    # Project all points onto the masked frame(s).
    voxel_frames = masked_frames[CAMERA_INDEX_MASK, XSS, YSS]

    # Only pixels that are visible in all cameras count as actual.
    combined_voxel_frames = np.all(voxel_frames, axis=0)
    visible_voxels = VOXEL_GRID[combined_voxel_frames]

    voxel_colours = np.empty((1,))
    if do_colour:
        visible_xss = XSS[CAMERA_INDEX_MASK, combined_voxel_frames]
        visible_yss = YSS[CAMERA_INDEX_MASK, combined_voxel_frames]

        coloured_videos = CACHED_ORIGINAL_VIDEOS[:, frame_id]
        voxel_colours = coloured_videos[
            CAMERA_INDEX_MASK, 
            visible_xss, 
            visible_yss
        ]
        voxel_colours = np.mean(voxel_colours, axis=0)

    return \
        visible_voxels, \
        voxel_colours if do_colour else \
            np.full((len(visible_voxels), 3), [255, 0, 0], dtype=np.uint8)

def get_cam_positions()-> tuple[cv2.typing.MatLike, list[list[float]]]:
    """
    Gets the voxel-world position of the cameras.

    First uses the rotation and translation vectors of the cameras to
    determine the location of the camera compared to the chessboard.
    Then uses the origin defined in `origin` and the `VOXEL_SIZE` to
    translate the camera position to it's voxel-world position. Finally
    switches the Y and Z axis to match the glm world.

    :returns: An array containing the in-voxel-world location of the
        cameras and a list containing color values for the cameras.
    :rtype: tuple[cv2.typing.MatLike, list[list[float]]]
    """
    cam_positions = []
    for camera in CAMERA_CONFIGS:
        rmat, _ = cv2.Rodrigues(camera.rvec)
        camera_position = np.dot(-rmat.T, camera.tvec)
        replace = camera_position.flatten() - ORIGIN
        in_grid = replace / VOXEL_SIZE
        in_grid *= block_size
        in_grid[2] = in_grid[2] * -1 + config['world_height']
        in_grid = in_grid - np.array(
            [config['world_width'] / 2, config['world_depth'] / 2, 0], 
            dtype=np.float32
        )
        cam_positions.append(in_grid)
    cam_positions = np.array(cam_positions)[:, [0, 2, 1]]
    return cam_positions, [[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0, 1, 0]]


def get_cam_rotation_matrices()-> list[glm.mat4]:
    """
    Gets the voxel-world rotation of the cameras.

    Before making the rotation matrix an homogenous matrix translate the
    rotation matrix so it's XYZ columns become ordered as YZX to match 
    the voxel-world.

    :returns: A list containing glm mat4 4x4 rotations of the cameraas
    :rtype: list[glm.mat4]
    """
    # Reorder columns so XYZ are now YZX
    P = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    cam_rotations = []
    for camera in CAMERA_CONFIGS:
        rmat, _ = cv2.Rodrigues(camera.rvec)
        
        R_gl = rmat @ P
        
        rotation = np.zeros((4, 4))
        rotation[:3, :3] = R_gl
        rotation[3, 3] = 1
        
        cam_rotations.append(glm.mat4(rotation))
    
    return cam_rotations

