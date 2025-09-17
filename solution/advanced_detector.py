# solution/advanced_detector.py

import os
import cv2
import numpy as np

def _ensure_outputs_dir() -> str:
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _stabilize_to_prev(prev_gray: np.ndarray, curr_bgr: np.ndarray):
    """
    Align current frame to the previous (grayscale) frame using sparse LK flow
    + robust affine estimation. Returns (stabilized_bgr, curr_gray, ok_flag).
    """
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    # Feature detect on previous stabilized gray
    pts_prev = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8
    )
    if pts_prev is None or len(pts_prev) < 10:
        return curr_bgr, curr_gray, False

    # Track features to the current frame
    pts_curr, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts_prev, None, winSize=(21, 21), maxLevel=3
    )
    if pts_curr is None:
        return curr_bgr, curr_gray, False

    st = st.reshape(-1)
    good_prev = pts_prev[st == 1].reshape(-1, 2)
    good_curr = pts_curr[st == 1].reshape(-1, 2)
    if len(good_prev) < 10:
        return curr_bgr, curr_gray, False

    # Affine (rotation/scale/translation) with RANSAC
    M, _ = cv2.estimateAffinePartial2D(
        good_curr, good_prev, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None:
        return curr_bgr, curr_gray, False

    H, W = prev_gray.shape
    stabilized_bgr = cv2.warpAffine(
        curr_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    stabilized_gray = cv2.warpAffine(
        curr_gray, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return stabilized_bgr, stabilized_gray, True

def _clean_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Clean MOG2 mask (0=bg, 127=shadow, 255=fg) → binary 0/255 and denoise.
    """
    mask = raw_mask.copy()
    # Remove shadows
    mask[mask == 127] = 0
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Morphology: open -> close -> (light) dilate
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)
    return mask

def _draw_overlay_and_boxes(frame_bgr: np.ndarray, mask: np.ndarray, min_area=250) -> np.ndarray:
    """
    Draw a green semi-transparent overlay where mask==255 and rectangles
    around connected components larger than min_area.
    """
    out = frame_bgr.copy()

    # Soft overlay
    overlay = out.copy()
    overlay[mask > 0] = (0, 255, 0)
    out = cv2.addWeighted(out, 0.75, overlay, 0.25, 0)

    # Bounding boxes
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return out

def advanced_motion_detector(video_path: str) -> str:
    """
    Implements a robust background subtraction method to detect motion in a video.

    Pipeline:
      1) Per-frame stabilization (affine) to suppress camera motion.
      2) Adaptive background modeling with MOG2 (shadows removed).
      3) Morphological cleanup + area filtering.
      4) Overlay + bounding boxes written to outputs/advanced_<name>.mp4.

    Returns:
        str: Path to the written output video.
    """
    out_dir = _ensure_outputs_dir()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"advanced_{base}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))

    # Background subtractor (tune history/varThreshold if needed)
    backsub = cv2.createBackgroundSubtractorMOG2(
        history=400, varThreshold=25, detectShadows=True
    )

    prev_gray = None
    warmup = 60  # faster learning at the start, slower after

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if prev_gray is None:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stabilized = frame
        else:
            stabilized, curr_gray, ok_stab = _stabilize_to_prev(prev_gray, frame)
            # keep the stabilized gray as the next reference if alignment succeeded
            prev_gray = curr_gray if ok_stab else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Adaptive learning rate: quick warm-up, then conservative updates
        lr = 0.2 if t < warmup else 0.01
        raw_mask = backsub.apply(stabilized, learningRate=lr)

        mask = _clean_mask(raw_mask)
        vis  = _draw_overlay_and_boxes(stabilized, mask, min_area=250)

        writer.write(vis)
        t += 1

    writer.release()
    cap.release()
    print(f"[advanced_motion_detector] wrote: {out_path}")
    return out_path

if __name__ == '__main__':
    # Example quick test (uncomment to run directly):
    # advanced_motion_detector('data/deer.mp4')
    pass
