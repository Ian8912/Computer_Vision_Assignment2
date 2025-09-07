# solution/motion_detector.py

import cv2
import numpy as np
from .helpers import parse_frame_name, make_frame_name, read_gray

def generate_motion_mask(frame_path: str, k: int, threshold: int) -> np.ndarray:
    
    MIN_FRAME = 0
    MAX_FRAME = 124

    # --- YOUR CODE GOES HERE ---

    # 1. Parse the frame path to get the directory and frame number `t`.
    #    You can use the provided `parse_frame_name` function.

    # 2. Determine the file paths for frame `t-k` and `t+k`.
    #    You can use the provided `make_frame_name` function.
    #    Handle edge cases: if `t-k` is less than MIN_FRAME, clamp it to MIN_FRAME.
    #    If `t+k` is greater than MAX_FRAME, clamp it to MAX_FRAME.

    # 3. ...

    # Replace this with your final mask
    motion_mask = np.zeros((1, 1), dtype=np.uint8) 
    return motion_mask


def clean_and_find_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    
    # --- YOUR CODE GOES HERE ---
    
    return (0, 0, 0, 0) # Replace this with your calculated bounding box (top, bottom, left, right)