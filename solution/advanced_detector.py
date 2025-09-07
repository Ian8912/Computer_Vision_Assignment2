# solution/advanced_detector.py

import cv2
import numpy as np

def advanced_motion_detector(video_path: str):
    """
    Implements a robust background subtraction method to detect motion in a video.

    This function should process a video file, detect motion using a method more
    advanced than simple frame differencing, and display or
    save the result.

    Args:
        video_path (str): The path to the input video file.
    """
    # --- YOUR EXTRA CREDIT CODE GOES HERE ---

    # You will need to open the video, loop through its frames, apply the
    # background subtractor, and possibly do some cleaning on the resulting mask.
    
    print(f"Processing video: {video_path}")
    print("Extra credit function is not yet implemented.")

    pass

if __name__ == '__main__':
    # You can use this section to test your advanced detector.
    # Example:
    # video_to_process = 'data/deer.mp4' # Assuming you create a video from the failure case frames
    # advanced_motion_detector(video_to_process)
    pass