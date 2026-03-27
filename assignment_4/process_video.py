"""
Annotate a video with YOLO predictions.

Loads a previously trained YOLOv1 model and processes a video file in
batches of frames, drawing bounding box predictions on each frame and
writing the result to a new video file.

Usage:
    python predict_video.py --model path/to/model.pth --video path/to/video.mp4

    # Full options:
    python predict_video.py \
        --model  assignment_4/models/base_model.pth \
        --video  input.mp4 \
        --output annotated_output.mp4 \
        --img-size 112 \
        --grid-size 7 \
        --conf-threshold 0.35 \
        --batch-size 16
"""

import argparse
import copy

import cv2
import numpy as np
import torch
from torchvision import transforms

from create_logger import create_logger
from decode import decode_predictions
from visualise import denormalise, draw_boxes
from yolov1_resnet import YOLOv1ResNet


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["cat", "dog"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_transform(img_size: int) -> transforms.Compose:
    """
    Build the inference transform pipeline.

    Mirrors the transform used during training: resize, convert to
    tensor, apply ImageNet normalisation.

    :param img_size: Target square size for the model input.
    :type img_size: int
    :returns: Composed transform.
    :rtype: transforms.Compose
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def frames_to_tensor(
    frames: list[np.ndarray],
    transform: transforms.Compose,
    device: str,
) -> torch.Tensor:
    """
    Convert a list of BGR uint8 frames to a normalised model-ready
    batch tensor.

    :param frames: Raw frames grabbed from cv2, in BGR uint8 format.
    :type frames: list[np.ndarray]
    :param transform: Transform pipeline to apply to each frame.
    :type transform: transforms.Compose
    :param device: Device to move the tensor to.
    :type device: str
    :returns: Float tensor of shape (N, C, img_size, img_size).
    :rtype: torch.Tensor
    """
    tensors = []
    for frame in frames:
        # cv2 reads BGR — convert to RGB before feeding the transform.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensors.append(transform(rgb))
    return torch.stack(tensors).to(device)


def annotate_frame(
    frame: np.ndarray,
    prediction_data: tuple,
    conf_threshold: float,
) -> np.ndarray:
    """
    Draw bounding boxes onto a single BGR frame.

    The frame is first converted to float [0, 1] RGB (matching what
    ``draw_boxes`` expects from ``visualise.py``), boxes are drawn, and
    the result is converted back to uint8 BGR for writing to video.

    :param frame: Original BGR uint8 frame from cv2.
    :type frame: np.ndarray
    :param prediction_data: Decoded model output tuple 
        (corrected_x, corrected_y, w, h, objectness, classes) all
        sliced to a single image (no batch dimension).
    :type prediction_data: tuple
    :param conf_threshold: Only detections above this objectness score
        are drawn.
    :type conf_threshold: float
    :returns: Annotated BGR uint8 frame.
    :rtype: np.ndarray
    """
    # Convert to float RGB [0, 1] — draw_boxes works in that space.
    rgb_float = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    annotated = draw_boxes(rgb_float, prediction_data, conf_threshold, CLASS_NAMES)

    # Convert back to uint8 BGR for VideoWriter.
    bgr_uint8 = cv2.cvtColor(
        (annotated * 255).clip(0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    return bgr_uint8


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_video(
    model_path: str,
    video_path: str,
    output_path: str,
    img_size: int,
    grid_size: int,
    conf_threshold: float,
    batch_size: int,
) -> None:
    """
    Load a trained model and produce an annotated copy of a video.

    Reads the input video in batches of ``batch_size`` frames, runs the
    model on each batch, draws predictions, and immediately writes the
    annotated frames to the output video — keeping peak memory use low.

    :param model_path: Path to the ``.pth`` model file.
    :type model_path: str
    :param video_path: Path to the source video file.
    :type video_path: str
    :param output_path: Where to write the annotated video.
    :type output_path: str
    :param img_size: Square size to resize frames to before inference.
    :type img_size: int
    :param grid_size: YOLO grid size (must match what the model was 
        trained with).
    :type grid_size: int
    :param conf_threshold: Objectness confidence threshold for drawing
        boxes.
    :type conf_threshold: float
    :param batch_size: Number of frames to process per model forward 
        pass.
    :type batch_size: int
    """
    logger = create_logger("predict_video")

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = YOLOv1ResNet.load(model_path, logger)
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from: {model_path}")

    # ── Transform ─────────────────────────────────────────────────────────────
    transform = build_transform(img_size)

    # ── Video reader ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    orig_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Video: {orig_w}x{orig_h} @ {fps:.2f} fps | "
        f"{total_frames} frames total"
    )

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer at: {output_path}")

    logger.info(f"Writing annotated video to: {output_path}")

    # ── Main loop — process in batches ────────────────────────────────────────
    frame_buffer: list[np.ndarray] = []
    frames_processed = 0

    def _flush_batch(buffer: list[np.ndarray]) -> None:
        """Run inference on a buffer of frames and write them out."""
        batch_tensor = frames_to_tensor(buffer, transform, device)

        with torch.no_grad():
            output = model(batch_tensor)

        # Reshape flat output to cube: (N, grid, grid, 7)
        cube = output.view(-1, grid_size, grid_size, 7)
        decoded = decode_predictions(cube)
        # decoded is a 6-tuple of tensors, each with a leading batch dim.

        for i, frame in enumerate(buffer):
            # Slice each decoded tensor down to a single image.
            frame_prediction = tuple(t[i] for t in decoded)
            annotated = annotate_frame(frame, frame_prediction, conf_threshold)
            writer.write(annotated)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame)

        if len(frame_buffer) == batch_size:
            _flush_batch(frame_buffer)
            frame_buffer.clear()
            frames_processed += batch_size
            logger.debug(
                f"Processed {frames_processed}/{total_frames} frames "
                f"({100 * frames_processed / max(total_frames, 1):.1f}%)"
            )

    # Process any remaining frames that didn't fill a complete batch.
    if frame_buffer:
        _flush_batch(frame_buffer)
        frames_processed += len(frame_buffer)

    cap.release()
    writer.release()
    logger.info(
        f"Done. {frames_processed} frames written to '{output_path}'."
    )


if __name__ == "__main__":
    process_video(
        model_path="assignment_4/models/resnet_model.pth",
        video_path="assignment_4/data/cool_video_cropped.mp4",
        output_path="assignment_4/output/processed_video.mp4",
        img_size=224,
        grid_size=7,
        conf_threshold=0.5,
        batch_size=16,
    )