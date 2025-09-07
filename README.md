# Assignment 2: Motion Detection - Implementation and Analysis

**Objective:** To implement a foundational computer vision algorithm, analyze its behavior under different parameters, understand its limitations, and optionally use AI tools to explore more advanced solutions.

## 1. Introduction

In this assignment, you will build a motion detector from the ground up. This is a classic computer vision task that introduces key concepts like image arithmetic, thresholding, morphological operations, and connected components.

The assignment is divided into three parts:
1.  **Core Implementation:** You will implement the core functions of the motion detector.
2.  **Experimental Analysis:** You will use your code to run experiments, analyze the results, and document your findings in a report.
3.  **Extra Credit:** You will research and implement a more advanced background subtraction technique. This is **optional**.

You are encouraged to use generative AI (like ChatGPT, Copilot, etc.) as a tool to help you write code. However, the analysis and justifications in your `REPORT.md` must be your own, based on the results you generate.

## 2. Setup and Workflow

1.  **Clone the Repository:** Clone this repository to your local machine or open it in a Codespace.
2.  **Install Dependencies:** Set up your Python environment and install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Locate the Files:**
    *   The primary image sequence is in `data/walkstraight/`.
    *   The sequence for the analysis task is in `data/failure_case/`.
    *   You will write your **core implementation** in `solution/motion_detector.py`.
    *   You will write your **extra credit** code in `solution/advanced_detector.py`.
    *   You will create a `REPORT.md` file in the root directory for your analysis.
4.  **Commit and Push:** As you complete parts of the assignment, commit and push your changes to see your autograding score for Part 1.
    ```bash
    git add .
    git commit -m "Completed Part 1 of motion detector"
    git push
    ```

## 3. Part 1: Core Implementation (70 Points)

Your task is to implement two functions in the `solution/motion_detector.py` file. These functions form the core pipeline of our simple motion detector.

---

### Function 1: `generate_motion_mask(frame_path, k, threshold)`

This function will perform three-frame differencing to create a raw motion mask.

*   **Inputs:**
    *   `frame_path` (str): The full path to the current frame (e.g., `'data/walkstraight/frame0062.tif'`).
    *   `k` (int): The frame offset for differencing (e.g., `10`).
    *   `threshold` (int): The intensity threshold for binarization (e.g., `10`).
*   **Implementation Steps:**
    1.  Parse the `frame_path` to get the directory and frame number. You are provided with helper functions (`parse_frame_name`, `make_frame_name`) for this.
    2.  Determine the file paths for frame `t-k` and frame `t+k`. Handle edge cases where `t-k` or `t+k` go out of bounds (0-124 for the `walkstraight` sequence).
    3.  Read the three frames (`t`, `t-k`, `t+k`) in **grayscale**.
    4.  Calculate the two absolute differences: `diff1 = abs(frame_t - frame_{t-k})` and `diff2 = abs(frame_t - frame_{t+k})`.
    5.  Compute the per-pixel minimum of the two differences to get the final motion image.
    6.  Apply a binary threshold to the motion image.
*   **Return:** The function must **return** the final binary motion mask as a `uint8` NumPy array (with values of 0 and 255).

---

### Function 2: `clean_and_find_bbox(mask)`

This function takes a noisy binary mask, cleans it, and finds the bounding box of the most significant motion.

*   **Input:**
    *   `mask` (np.ndarray): A binary motion mask (the output from the previous function).
*   **Implementation Steps:**
    1.  **Clean the mask:** Apply a **morphological opening** operation to the mask to remove small noise specks. Use a 3x3 rectangular kernel.
    2.  **Find connected components:** Use `cv2.connectedComponentsWithStats` to find all distinct objects in the cleaned mask.
    3.  **Isolate the largest component:** Identify the largest component by area (ignoring the background, which is label 0).
    4.  **Calculate bounding box:** Determine the coordinates `(top, bottom, left, right)` for the bounding box of this largest component.
*   **Return:** The function must **return** the bounding box as a tuple of four integers: `(top_row, bottom_row, left_column, right_column)`. If no components are found (other than the background), it should return `(0, 0, 0, 0)`.

## 4. Part 2: Experimental Analysis (30 Points)

For this part, you will create a new file named `REPORT.md` in the root of your repository. You will use the functions you built in Part 1 to conduct experiments and answer the following questions.

---

### Section 1: The "k" Parameter Experiment

Use your `generate_motion_mask` function on the frame `'data/walkstraight/frame0062.tif'` with a fixed `threshold=10`. Generate the motion masks for `k = 1`, `k = 10`, and `k = 25`.

1.  **Include the three resulting mask images** in your report.
2.  **Answer the following questions:**
    *   **For k=1:** Why is the resulting mask so sparse and noisy? What does this tell you about the amount of motion between two consecutive frames in this video?
    *   **For k=25:** You will likely see a "ghosting" or "doubled" silhouette. Explain what information from which specific frames (`t-25`, `t`, `t+25`) contributes to this visual artifact.
    *   **Justification:** Based on these results, explain why `k=10` provides a good balance between sensitivity and clarity for this specific video.

---

### Section 2: Failure Case Analysis

Run your full pipeline (both `generate_motion_mask` and `clean_and_find_bbox`) on the frame `'data/failure_case/frame0015.tif'` using the recommended parameters (`k=10`, `threshold=10`).

1.  **Include the final image** showing the detected bounding box on this failure case frame. You will need to write a small script to generate this. Save the output image in the `output/` folder.
2.  **Answer the following questions:**
    *   The simple frame differencing algorithm is built on a key assumption about the scene. What is this assumption?
    *   Based on what you see in the failure case video, explain how this assumption is violated and why it causes the algorithm to fail.

## 5. Part 3: Extra Credit (Up to 20 Points)

The simple frame differencing method is not robust to real-world challenges like illumination changes or dynamic backgrounds. For extra credit, research and implement a more advanced background subtraction technique in the file `solution/advanced_detector.py`.

*   **Task:** Implement the function `advanced_motion_detector(video_path)`, which takes a path to a video file and produces an output video showing the detected motion.
*   **Recommended Technique:** A great method to investigate is the **Running Gaussian Average (MOG2)**, available in OpenCV as `cv2.createBackgroundSubtractorMOG2()`. You are highly encouraged to use an LLM for research and to help generate the initial code for this.
*   **Deliverables:**
    1.  Your implementation in `solution/advanced_detector.py`.
    2.  A new section in `REPORT.md` titled "Extra Credit: Advanced Motion Detection".
    3.  In this section, you must include:
        *   **The Prompt:** The exact prompt(s) you used with an LLM that helped you solve this.
        *   **The Result:** A side-by-side comparison (GIFs or still images) showing the output of your simple detector vs. your advanced detector on the `failure_case` data. Save your output GIF in the `output/` folder.
        *   **Analysis:** A paragraph explaining *why* the MOG2 method is more robust than simple frame differencing for this specific failure case.

## 6. Deliverables Summary

-   **Code:**
    -   Completed `solution/motion_detector.py`.
    -   (Optional) Completed `solution/advanced_detector.py`.
-   **Report:**
    -   A `REPORT.md` file in the root directory containing your analysis for Part 2 and (optionally) Part 3.
-   **Output Files:**
    -   The output image from the failure case analysis (`output/failure_case_output.png`).
    -   (Optional) The output GIF from your advanced detector (`output/advanced_detector_output.gif`).
