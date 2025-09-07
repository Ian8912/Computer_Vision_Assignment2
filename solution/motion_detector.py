# solution/motion_detector.py

import cv2
import numpy as np
from .helpers import parse_frame_name, make_frame_name, read_gray

def generate_motion_mask(video_path: str, t: int, k: int, threshold: int, blur_ksize: int) -> np.ndarray:

    # --- YOUR CODE GOES HERE ---

    # Replace this with your final mask
    motion_mask = np.zeros((1, 1), dtype=np.uint8) 
    return motion_mask


def clean_and_find_bbox(mask: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    
    # --- YOUR CODE GOES HERE ---

    bbox = (0, 0, 0, 0)  # Replace with actual bounding box
    cleaned_mask_with_bbox = np.zeros_like(mask)  # Replace with actual cleaned mask with bbox drawn
    
    return (bbox, cleaned_mask_with_bbox)