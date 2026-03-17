import cv2
import numpy as np
import torch


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

    object_confidence, corrected_x, corrected_y, w, h, predicted_class = \
        prediction_data
    
    # Filter on indices that contain objects with high enough threshold.
    valid_cells = (object_confidence > confidence_threshold)

    object_confidence = object_confidence[valid_cells]
    corrected_x = corrected_x[valid_cells]
    corrected_y = corrected_y[valid_cells]
    w = w[valid_cells]
    h = h[valid_cells]
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
        color = (31, 119, 180) if cls == 0 else (255, 127, 14)
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
