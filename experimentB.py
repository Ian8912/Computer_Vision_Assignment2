import os
import cv2

from solution.motion_detector import generate_motion_mask

# Use the frame-sequence stem (Option 1)
VIDEO = "data/walkstraight/frame"
T = 62
K = 10       # keep same k as your baseline
THR = 5

def put_label(gray, text):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(bgr, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return bgr

def main():
    os.makedirs("outputs", exist_ok=True)

    # No blur (blur_ksize=1 means “skip blur” in our function)
    mask_nb = generate_motion_mask(VIDEO, t=T, k=K, threshold=THR, blur_ksize=1)
    cv2.imwrite("outputs/expB_mask_blur1_thr5.png", mask_nb)

    # Stronger blur
    mask_b7 = generate_motion_mask(VIDEO, t=T, k=K, threshold=THR, blur_ksize=7)
    cv2.imwrite("outputs/expB_mask_blur7_thr5.png", mask_b7)

    # Side-by-side panel
    left  = put_label(mask_nb, "blur_ksize=1 (no blur), thr=5")
    right = put_label(mask_b7, "blur_ksize=7, thr=5")
    panel = cv2.hconcat([left, right])
    cv2.imwrite("outputs/expB_side_by_side.png", panel)

if __name__ == "__main__":
    main()
