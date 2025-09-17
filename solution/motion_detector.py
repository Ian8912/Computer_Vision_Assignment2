# solution/motion_detector.py
import os
import cv2
import numpy as np
from .helpers import make_frame_name, read_gray  # keep these; used when reading a frame sequence

# ---------- helpers ----------
def _read_gray_from_video(video_path: str, idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if idx < 0 or idx >= total:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def _resolve_frame_stem(path_or_stem: str, t: int) -> str:
    """
    Accepts:
      - '.../name.mp4'  -> try '.../name/frame'
      - '.../name' (dir)-> try '.../name/frame' then '.../name'
      - '.../name/frame' (already stem) -> use as-is
    Returns a stem such that make_frame_name(stem, t) exists.
    """
    base, ext = os.path.splitext(path_or_stem)
    candidates = []
    if ext.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        return ""  # signal: use video mode
    if os.path.isdir(path_or_stem):
        candidates = [os.path.join(path_or_stem, "frame"), path_or_stem]
    else:
        candidates = [path_or_stem]

    for stem in candidates:
        test_path = make_frame_name(stem, t)
        if os.path.exists(test_path):
            return stem
    # If nothing matched, return original (will likely fail and show the path in the error)
    return candidates[-1]

def _read_gray_from_sequence(stem: str, idx: int):
    path = make_frame_name(stem, idx)
    img = read_gray(path)   # your helper reads a single image path
    return img

# ---------- main API ----------
def generate_motion_mask(video_path: str, t: int, k: int, threshold: int, blur_ksize: int) -> np.ndarray:
    """
    Three-frame differencing over frames (t-k, t, t+k) → binary mask {0,255}.
    Works with either:
      - MP4 path (reads frames via cv2.VideoCapture), or
      - frame-sequence stem like 'data/walkstraight/frame' (uses make_frame_name + read_gray).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not (0 <= threshold <= 255):
        raise ValueError("threshold must be in [0, 255]")

    # Decide mode: video or sequence
    base, ext = os.path.splitext(video_path)
    use_video = ext.lower() in {".mp4", ".avi", ".mov", ".mkv"}

    if use_video:
        g_prev = _read_gray_from_video(video_path, t - k)
        g_curr = _read_gray_from_video(video_path, t)
        g_next = _read_gray_from_video(video_path, t + k)
    else:
        stem = _resolve_frame_stem(video_path, t)
        g_prev = _read_gray_from_sequence(stem, t - k)
        g_curr = _read_gray_from_sequence(stem, t)
        g_next = _read_gray_from_sequence(stem, t + k)

    if any(x is None for x in (g_prev, g_curr, g_next)):
        raise IndexError(f"One or more frames not available for indices {(t-k, t, t+k)} in '{video_path}'.")

    # Optional denoise
    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1  # Gaussian kernel must be odd
        g_prev = cv2.GaussianBlur(g_prev, (blur_ksize, blur_ksize), 0)
        g_curr = cv2.GaussianBlur(g_curr, (blur_ksize, blur_ksize), 0)
        g_next = cv2.GaussianBlur(g_next, (blur_ksize, blur_ksize), 0)

    # Differencing & combine
    d1 = cv2.absdiff(g_curr, g_prev)
    d2 = cv2.absdiff(g_next, g_curr)
    D  = cv2.min(d1, d2)

    # Threshold to binary
    _, mask = cv2.threshold(D, threshold, 255, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)

def clean_and_find_bbox(mask: np.ndarray):
    """
    Clean a noisy binary motion mask and return the largest component's bbox,
    plus the cleaned mask with the bbox drawn on it.
    Returns: ((top, bottom, left, right), cleaned_with_box)
    """
    if mask is None:
        raise ValueError("mask is None")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Normalize to {0,255}
    if mask.max() == 1:
        mask = (mask * 255).astype(np.uint8)

    # Morphological cleanup
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    # Connected components on 0/1 image
    fg = (cleaned > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    if num_labels <= 1:
        # Nothing found; return the cleaned mask as-is
        return ((0, 0, 0, 0), cleaned)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))

    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])

    top_row = y
    bottom_row = y + h - 1
    left_col = x
    right_col = x + w - 1
    bbox = (top_row, bottom_row, left_col, right_col)

    # Keep only the largest component & draw box
    largest_mask = np.zeros_like(cleaned, dtype=np.uint8)
    largest_mask[labels == largest_idx] = 255
    out = largest_mask.copy()
    cv2.rectangle(out, (left_col, top_row), (right_col, bottom_row), 255, 2)

    return (bbox, out)
