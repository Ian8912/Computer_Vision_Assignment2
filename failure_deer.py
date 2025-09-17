import os
import cv2
import numpy as np

from solution.motion_detector import generate_motion_mask  # if you exported frames instead (see note)
from solution.motion_detector import clean_and_find_bbox  # or wherever you placed it

VIDEO_MP4 = "data/deer.mp4"
T   = 260        
K   = 10          # temporal offset
BLR = 5          # blur kernel (odd)
THR = 6         # threshold

OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

def read_gray_from_video(path, idx):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if idx < 0 or idx >= total:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), frame  # return gray + original BGR

def main():
    # Build mask from the MP4 directly (you can swap to your own generate_motion_mask()
    # if you first export frames to data/deer/frame#####.tif and pass the stem).
    mask = generate_motion_mask(VIDEO_MP4, T, K, THR, BLR)

    # Clean + bbox on the mask
    bbox, mask_with_box = clean_and_find_bbox(mask)

    # Also draw bbox on the ORIGINAL frame t (for the report image)
    _, frame_bgr = read_gray_from_video(VIDEO_MP4, T)
    if frame_bgr is None:
        raise RuntimeError("Could not read frame for overlay.")
    top, bot, left, right = bbox
    if (top, bot, left, right) != (0, 0, 0, 0):
        cv2.rectangle(frame_bgr, (left, top), (right, bot), (0, 255, 0), 2)

    # Save both views
    cv2.imwrite(f"{OUT_DIR}/deer_mask_clean_bbox.png", mask_with_box)
    cv2.imwrite(f"{OUT_DIR}/deer_bbox_on_frame.png", frame_bgr)

    print("Saved:")
    print(f" - {OUT_DIR}/deer_mask_clean_bbox.png")
    print(f" - {OUT_DIR}/deer_bbox_on_frame.png")
    print(f"BBox: {bbox}")

if __name__ == "__main__":
    main()