import copy
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from plum import dispatch
from torch.utils.data import DataLoader
from typing import Any

from decode import decode_predictions


COLOUR_CAT = (255, 127, 14)
COLOUR_DOG = (31, 119, 180)
COLOUR_CAT_NORMALISED = tuple(np.array(COLOUR_CAT) / 255)
COLOUR_DOG_NORMALISED = tuple(np.array(COLOUR_DOG) / 255)


def draw_boxes(
    image: cv2.typing.MatLike, 
    prediction_data: tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ],
    confidence_threshold: float,
    class_names: list[str],
)-> cv2.typing.MatLike:
    """
    Display bounding boxes on top of image.
    
    :param image: Image to draw boxes onto.
    :type image: cv2.typing.MatLike
    :param prediction_data: Decoded output from model.
    :type prediction_data: tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ]
    :param confidence_threshold: Only boxes above this threshold are 
        displayed.
    :type confidence_threshold: float
    :param class_names: Names of the classes.
    :type class_names: list[str]
    :return: Image with bounding boxes drawn onto it.
    :rtype: cv2.typing.MatLike
    """
    image = image.copy()
    img_h, img_w = image.shape[:2]

    corrected_x, corrected_y, w, h, object_confidence, classes = \
        prediction_data
    predicted_class = torch.argmax(classes, dim=-1) 
    
    # Filter on indices that contain objects with high enough threshold.
    valid_cells = (object_confidence > confidence_threshold)

    corrected_x = corrected_x[valid_cells]
    corrected_y = corrected_y[valid_cells]
    w = w[valid_cells]
    h = h[valid_cells]
    object_confidence = object_confidence[valid_cells]
    predicted_class = predicted_class[valid_cells]

    # Convert from relative size to pixel size.
    pixel_relative_x = corrected_x * img_w
    pixel_relative_y = corrected_y * img_h
    pixel_relative_w = w * img_w
    pixel_relative_h = h * img_h

    # Convert from centers to corners.
    x1 = (pixel_relative_x - pixel_relative_w / 2).int()
    y1 = (pixel_relative_y - pixel_relative_h / 2).int()
    x2 = (pixel_relative_x + pixel_relative_w / 2).int()
    y2 = (pixel_relative_y + pixel_relative_h / 2).int()

    for i in range(len(object_confidence)):
        cls = predicted_class[i].item()
        # Colours are friendly for colourblind people.
        color = COLOUR_CAT_NORMALISED if cls == 0 else COLOUR_DOG_NORMALISED
        label = f"{class_names[cls]} {object_confidence[i]:.2f}"

        # Bounding box.
        cv2.rectangle(
            image, 
            (x1[i].item(), y1[i].item()), 
            (x2[i].item(), y2[i].item()), 
            color, 
            2
        )
        ################### Label background & text. ###################
        (label_w, label_h), baseline = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            1
        )
        label_x = x1[i].item()
        label_y = y1[i].item() - 5

        # If label goes above the image, draw it inside the box top instead.
        if label_y - label_h - baseline < 0:
            label_y = y1[i].item() + label_h + baseline

        # If label goes off the right edge, shift it left.
        if label_x + label_w > img_w:
            label_x = img_w - label_w

        # Clamp to left edge.
        label_x = max(0, label_x)

        # Do some funky math to make the label background opaque.
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (label_x, label_y - label_h - baseline),
            (label_x + label_w, label_y + baseline),
            color,
            cv2.FILLED
        )
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.putText(
            image,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (1.0, 1.0, 1.0),
            1
        )

    return image


@dispatch
def visualise_batch(
    dataloader: DataLoader, 
    confidence_threshold: float,
    output_path: str
)-> None:
    """
    Visualise a single batch from a dataloader.

    :param dataloader: Dataloader to visualise.
    :type dataloader: Dataloader
    :param confidence_threshold: Confidence threshold for plotting.
    :type confidence_threshold: float
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    images, targets = next(iter(dataloader))

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]
            
    targets = decode_predictions(targets, dataloader.dataset.dataset.grid_size)
    visualise_batch(
        images, 
        targets, 
        axes, 
        confidence_threshold,
        dataloader.dataset.dataset.classes,
        output_path
    )

@dispatch
def visualise_batch(
    X: torch.Tensor, 
    y: torch.Tensor, 
    confidence_threshold: float,
    class_names: list[str],
    output_path: str
)-> None:
    """
    Visualise a single batch from a X and y.

    :param X: Batch of images.
    :type X: torch.Tensor.
    :param y: Batch of targets (cubed).
    :type y: torch.Tensor.
    :param y_hat: Batch of prediction targets (cubed).
    :type y_hat: torch.Tensor.
    :param confidence_threshold: Confidence threshold for plotting.
    :type confidence_threshold: float
    :param class_names: Names of all the possible classes.
    :type class_names: list[str]
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    fig, axes = plt.subplots(1, len(X), figsize=(15, 5))
    if len(X) == 1:
        axes = [axes]
            
    targets = decode_predictions(y, y.shape[1])
    visualise_batch(
        X, 
        targets, 
        axes, 
        class_names, 
        confidence_threshold, 
        output_path
    )

@dispatch
def visualise_batch(
    X: torch.Tensor, 
    y: torch.Tensor, 
    y_hat: torch.Tensor, 
    confidence_threshold: float,
    class_names: list[str],
    output_path: str
)-> None:
    """
    Plots the true labels on top, and the predictions on the bottom.

    :param X: Batch of images.
    :type X: torch.Tensor.
    :param y: Batch of targets (cubed).
    :type y: torch.Tensor.
    :param y_hat: Batch of prediction targets (cubed).
    :type y_hat: torch.Tensor.
    :param confidence_threshold: Confidence threshold for plotting.
    :type confidence_threshold: float
    :param class_names: Names of all the possible classes.
    :type class_names: list[str]
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    fig, axes = plt.subplots(2, len(X), figsize=(15, 5))
    targets_y = decode_predictions(y, y.shape[1])
    targets_y_hat = decode_predictions(y_hat, y.shape[1])

    for i, img in enumerate(X):
        pic = copy.deepcopy(img).cpu()
        pic = pic.permute(1, 2, 0).numpy()
        pic = draw_boxes(
            pic,
            tuple(t[i] for t in targets_y),
            confidence_threshold, 
            class_names
        ) 
        axes[0,i].imshow(pic)

        pic = copy.deepcopy(img).cpu()
        pic = pic.permute(1, 2, 0).numpy()
        pic = draw_boxes(
            pic,
            tuple(t[i] for t in targets_y_hat),
            confidence_threshold, 
            class_names
        ) 
        axes[1, i].imshow(pic)

        axes[0, i].axis('off')
        axes[1, i].axis('off')

    fig.text(0.01, 0.75, 'Ground Truth', va='center', ha='left',
             fontsize=13, fontweight='bold', rotation=90)
    fig.text(0.01, 0.25, 'Predictions', va='center', ha='left',
             fontsize=13, fontweight='bold', rotation=90)

    # Leave left margin for labels.
    plt.tight_layout(rect=[0.03, 0, 1, 1])  
    plt.savefig(output_path, bbox_inches='tight')

@dispatch
def visualise_batch(
    images: torch.Tensor, 
    targets: tuple[
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor, 
        torch.Tensor
    ],
    axes: Any,
    confidence_threshold: float,
    class_names: list[str],
    output_path: str
)-> None:
    """
    Visualise a single batch from a set of images and targets.

    CODE PARTIALLY PROVIDED IN ASSIGNMENT.

    :param images: Images to visualise.
    :type images: torch.Tensor
    :param targets: Target objects, unpacked cubes.
    :type targets: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    :param axes: axes to display onto.
    :type axes: Any
    :param confidence_threshold: Confidence threshold for plotting.
    :type confidence_threshold: float
    :param class_names: Names of all the possible classes.
    :type class_names: list[str]
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    for i, img in enumerate(images):
        pic = copy.deepcopy(img).cpu()
        pic = pic.permute(1, 2, 0).numpy()
        pic = draw_boxes(
            pic,
            tuple(t[i] for t in targets),
            confidence_threshold, 
            class_names
        ) 
        axes[i].imshow(pic)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
