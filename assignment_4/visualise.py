import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from torch.utils.data import DataLoader


COLOUR_CAT = (255, 127, 14)
COLOUR_DOG = (31, 119, 180)
COLOUR_CAT_NORMALISED = np.array(COLOUR_CAT) / 255
COLOUR_DOG_NORMALISED = np.array(COLOUR_DOG) / 255


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

    corrected_x, corrected_y, w, h, object_confidence, predicted_class = \
        prediction_data
    
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
        color = COLOUR_CAT if cls == 0 else COLOUR_DOG
        label = f"{class_names[cls]} {object_confidence[i]:.2f}"

        cv2.rectangle(
            image, 
            (x1[i].item(), y1[i].item()), 
            (x2[i].item(), y2[i].item()), 
            color, 
            2
        )
        cv2.putText(
            image, 
            label, 
            (x1[i].item(), y1[i].item() - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            1
        )
    return image


# Function to visualize a batch
def visualize_batch(dataloader: DataLoader)-> None:
    """
    Visualise a single batch from a dataloader.

    :param dataloader: Dataloader to visualise.
    :type dataloader: Dataloader
    """
    images, bboxes, labels = next(iter(dataloader))

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    if len(images) == 1:
        axes = [axes]

    for i, (img, bbox, label) in enumerate(zip(images, bboxes, labels)):
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        for box, lbl in zip(bbox, label):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle(
                (xmin, ymin), 
                xmax - xmin, 
                ymax - ymin,
                linewidth=2,
                edgecolor=COLOUR_CAT_NORMALISED if lbl.item() == 0 \
                    else COLOUR_DOG_NORMALISED, 
                facecolor='none'
            )
            axes[i].add_patch(rect)
            axes[i].text(
                xmin, ymin - 5, 
                f'{"Cat" if lbl.item() == 0 else "Dog"} (label={lbl.item()})', 
                color=COLOUR_CAT_NORMALISED if lbl.item() == 0 \
                    else COLOUR_DOG_NORMALISED, 
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.5)
            )
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()