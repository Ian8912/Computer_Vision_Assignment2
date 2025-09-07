# solution/helpers.py

import cv2
import numpy as np
import os

def parse_frame_name(frame_filename: str) -> tuple[str, int]:
    """
    Separates the directory name and frame number from a filename.
    e.g., "data/walkstraight/frame0062.tif" -> ("data/walkstraight/frame", 62)
    """
    # Find the last occurrence of 'frame'
    base = os.path.splitext(frame_filename)[0] # remove .tif
    dir_and_base = base[:-4] # remove the 4-digit number
    frame_string = base[-4:]
    frame = int(frame_string)
    
    return dir_and_base, frame

def make_frame_name(dir_base: str, frame: int) -> str:
    """
    Creates a frame filename given the base directory name and the frame number.
    e.g., ("data/walkstraight/frame", 62) -> "data/walkstraight/frame0062.tif"
    """
    frame_filename = f"{dir_base}{frame:04d}.tif"
    return frame_filename

def read_gray(frame_path: str) -> np.ndarray:
    """
    Reads an image from the specified file path and converts it to grayscale.
    Returns a black image of size 1x1 if the image cannot be read.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Warning: Could not read image at {frame_path}")
        return np.zeros((1, 1), dtype=np.uint8)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame