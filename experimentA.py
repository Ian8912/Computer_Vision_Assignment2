import os
import cv2
import numpy as np

from solution.motion_detector import generate_motion_mask

VIDEO = "data/walkstraight/frame"
T = 62
BLUR = 5
THR = 10

def main():
    os.makedirs("outputs", exist_ok=True)

    # k=1
    mask_k1 = generate_motion_mask(VIDEO, t=T, k=1, threshold=THR, blur_ksize=BLUR)
    cv2.imwrite("outputs/mask_k1.png", mask_k1)

    # baseline k=10
    mask_k10 = generate_motion_mask(VIDEO, t=T, k=10, threshold=THR, blur_ksize=BLUR)
    cv2.imwrite("outputs/mask_k10.png", mask_k10)

    # side-by-side panel
    # convert to BGR for labeling
    k1_bgr = cv2.cvtColor(mask_k1, cv2.COLOR_GRAY2BGR)
    k10_bgr = cv2.cvtColor(mask_k10, cv2.COLOR_GRAY2BGR)

    # add simple labels
    def put_label(img, text):
        out = img.copy()
        cv2.rectangle(out, (0,0), (img.shape[1], 32), (0,0,0), -1)
        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return out

    k1_bgr = put_label(k1_bgr, "k = 1, blur=5, thr=10")
    k10_bgr = put_label(k10_bgr, "k = 10 (baseline), blur=5, thr=10")

    panel = cv2.hconcat([k1_bgr, k10_bgr])
    cv2.imwrite("outputs/expA_side_by_side.png", panel)

if __name__ == "__main__":
    main()
