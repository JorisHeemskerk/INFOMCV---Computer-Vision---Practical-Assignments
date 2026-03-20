import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

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


def visualise_batch(dataloader: DataLoader, output_path: str)-> None:
    """
    Visualise a single batch from a dataloader.

    CODE PARTIALLY PROVIDED IN ASSIGNMENT.

    :param dataloader: Dataloader to visualise.
    :type dataloader: Dataloader
    :param output_path: Where to save the file to.
    :type output_path: str
    """
    images, targets = next(iter(dataloader))

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]
            
    targets = decode_predictions(targets, dataloader.dataset.dataset.grid_size)
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).numpy()
        img = draw_boxes(
            img,
            tuple(t[i] for t in targets),
            0, 
            dataloader.dataset.dataset.classes
        ) 
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_path)

# def visualise_batch(X: torch.Tensor, y: torch.Tensor, output_path: str)-> None:
#     """
#     :param X: Batch of images.
#     :param y: Batch of targets (cubed).
#     """

# def visualise_batch(
#     X: torch.Tensor, 
#     y: torch.Tensor, 
#     y_hat: torch.Tensor, 
#     output_path: str
# )-> None:
#     """
#     Plots the true labels on top, and the predictions on the bottom.

#     :param X: Batch of images.
#     :param y: Batch of targets (cubed).
#     :param y_hat: Batch of prediction targets (cubed).
#     """
