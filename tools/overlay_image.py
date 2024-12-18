import cv2
import numpy as np
from mask_display import mask_map, visualize_wb_mask
from contour_detector import getting_coordinates


def painter_borders(mask_unique: np.ndarray, image: np.ndarray):
    im_overlay = image.copy()
    for mask in mask_map(mask_unique):
        for box in getting_coordinates(mask):
            (x, y, w, h) = [v for v in box]
            cv2.rectangle(im_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return im_overlay
