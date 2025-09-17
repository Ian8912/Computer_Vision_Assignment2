# scripts/extra_compare.py
import os
import cv2
import numpy as np

from solution.motion_detector import generate_motion_mask, clean_and_find_bbox
from solution.advanced_detector import advanced_motion_detector

VIDEO = "data/deer.mp4"
# comparison window (you can tweak these)
START, END = 240, 320      # inclusive start, exclusive end
# simple baseline params
K, BLUR, THR = 3, 5, 8

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def read_bgr(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None

def get_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, W, H

def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def make_simple_video():
    fps, W, H = get_meta(VIDEO)
    vw = cv2.VideoWriter(os.path.join(OUT_DIR, "simple_deer.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    for t in range(START, END):
        frame = read_bgr(VIDEO, t)
        if frame is None: break

        # 3-frame differencing baseline
        mask = generate_motion_mask(VIDEO, t=t, k=K, threshold=THR, blur_ksize=BLUR)
        bbox, cleaned = clean_and_find_bbox(mask)

        # overlay for visualization
        overlay = frame.copy()
        overlay[cleaned > 0] = (0, 255, 0)
        vis = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

        if bbox != (0, 0, 0, 0):
            top, bot, left, right = bbox
            cv2.rectangle(vis, (left, top), (right, bot), (0, 255, 0), 2)

        vw.write(label(vis, f"Simple (k={K}, blur={BLUR}, thr={THR})"))
    vw.release()

def make_side_by_side():
    # run advanced detector over full video (writes outputs/advanced_deer.mp4)
    adv_path = advanced_motion_detector(VIDEO)
    # normalize name for convenience
    fixed_adv = os.path.join(OUT_DIR, "advanced_deer.mp4")
    if os.path.abspath(adv_path) != os.path.abspath(fixed_adv):
        try:
            os.replace(adv_path, fixed_adv)
        except Exception:
            fixed_adv = adv_path  # in case replace isn't allowed

    # ensure simple video exists
    if not os.path.exists(os.path.join(OUT_DIR, "simple_deer.mp4")):
        make_simple_video()

    s_cap = cv2.VideoCapture(os.path.join(OUT_DIR, "simple_deer.mp4"))
    a_cap = cv2.VideoCapture(fixed_adv)

    # align on the same window
    a_cap.set(cv2.CAP_PROP_POS_FRAMES, START)

    fps = a_cap.get(cv2.CAP_PROP_FPS) or s_cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok_s, s = s_cap.read()
    ok_a, a = a_cap.read()
    if not (ok_s and ok_a):
        s_cap.release(); a_cap.release()
        raise RuntimeError("Could not read frames for side-by-side.")

    H, W = a.shape[:2]
    out_w = W * 2
    vw = cv2.VideoWriter(os.path.join(OUT_DIR, "extra_side_by_side.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (out_w, H))

    # restart the advanced read from START (we advanced once above)
    a_cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    # simple video starts at its first frame which already corresponds to START..END
    s_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok_a, a = a_cap.read()
        ok_s, s = s_cap.read()
        if not (ok_a and ok_s):
            break

        a = label(a, "Advanced (Stabilized MOG2)")
        s = label(s, "Simple (3-frame diff)")
        panel = cv2.hconcat([s, a])
        vw.write(panel)

    s_cap.release(); a_cap.release(); vw.release()

    # optional: make a GIF if imageio is installed
    try:
        import imageio.v3 as iio
        frames_rgb = []
        cap = cv2.VideoCapture(os.path.join(OUT_DIR, "extra_side_by_side.mp4"))
        for _ in range(1500):
            ok, fr = cap.read()
            if not ok: break
            frames_rgb.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if frames_rgb:
            iio.imwrite(os.path.join(OUT_DIR, "extra_side_by_side.gif"),
                        frames_rgb, loop=0, fps=int(fps))
    except Exception:
        pass

if __name__ == "__main__":
    # build simple and side-by-side
    make_simple_video()
    make_side_by_side()
    print("Wrote:")
    print("  outputs/simple_deer.mp4")
    print("  outputs/advanced_deer.mp4")
    print("  outputs/extra_side_by_side.mp4")
    if os.path.exists("outputs/extra_side_by_side.gif"):
        print("  outputs/extra_side_by_side.gif")
