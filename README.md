[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lOrmPsQD)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=20353000)
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
    *   The primary video is `data/walkstraight.mp4` (converted from the `data/walkstraight/` frames).
    *   The video for the failure case analysis task is in `data/deer.mp4`.
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

### Function 1: `generate_motion_mask(video_path, t, k, threshold, blur_ksize)`

This function performs three-frame differencing on frames extracted from a video to create a raw motion mask.

*   **Inputs:**
    *   `video_path` (str): The full path to the video file (e.g., `'data/walkstraight.mp4'`).
    *   `t` (int): The frame index to process (e.g., `62`).
    *   `k` (int): The frame offset for differencing (e.g., `10`).
    *   `threshold` (int): The intensity threshold for binarization (e.g., `10`).
    *   `blur_ksize` (int): The kernel size for blurring (e.g., `5`).
*   **Implementation:**
    Use the methods described in the lecture and the example code in the notebook to implement this function. Apply blurring, differencing, and thresholding to the three frames to create the motion mask. You do **not** need to find the largest component or draw the bounding box at this point. Just return the final binary motion mask, including any possible noise artifacts.
*   **Return:** The function must **return** the final binary motion mask as a `uint8` NumPy array (with values of 0 and 255).

---

### Function 2: `clean_and_find_bbox(mask)`

This function takes a noisy binary mask, cleans it, and finds the bounding box of the most significant motion.

*   **Input:**
    *   `mask` (np.ndarray): A binary motion mask (the output from the previous function).
*   **Implementation:** Use the methods described in the lecture and the example code in the notebook (largeset connected component, morphological operations, bounding box) to implement this function.
*   **Return:** Return: The function must return a tuple containing two items:
    1. The bounding box as a tuple of four integers: (top_row, bottom_row, left_column, right_column).
    2. The cleaned binary mask with the bounding box drawn on it as a uint8 NumPy array.

    (e.g., `((top_row, bottom_row, left_column, right_column), cleaned_mask_with_bbox)`)

## 4. Part 2: Experimental Analysis (30 Points)

For this part, you will create a new file named `REPORT.md` in the root of your repository. You will use the functions you built in Part 1 to conduct experiments and answer the following questions.

---

### Section 1: The "k" Parameter Experiment

Use your `generate_motion_mask` function on the video `data/walkstraight.mp4` at frame index `t=62` with a fixed `threshold=10`. Generate the motion masks for `k = 1`, `k = 10`, and `k = 25`.

1.  **Include the three resulting mask images** in your report.
2.  **Answer the following questions:**
    *   **For k=1:** Why is the resulting mask so sparse and noisy? What does this tell you about the amount of motion between two consecutive frames in this video?
    *   **For k=25:** You will likely see a "ghosting" or "doubled" silhouette. Explain what information from which specific frames (`t-25`, `t`, `t+25`) contributes to this visual artifact.
    *   **Justification:** Based on these results, explain why `k=10` provides a good balance between sensitivity and clarity for this specific video.

---

### Section 2: Failure Case Analysis

Run your full pipeline (both `generate_motion_mask` and `clean_and_find_bbox`) on the frame `'data/deer.mp4'` for any frame between `240` and `320` with any parameters you want. Try to find the parameters that work best for this case.

1.  **Include the final image** showing the detected bounding box on this failure case frame. You will need to write a small script to generate this. Save the output image in the `output/` folder.
2.  **Answer the following questions:**
    *   The simple frame differencing algorithm is built on a key assumption about the scene. What is this assumption?
    *   Based on what you see in the failure case video, explain how this assumption is violated and why it causes the algorithm to fail.

## 5. Part 3: Extra Credit (Up to 20 Points)

The simple frame differencing method is not robust to real-world challenges like illumination changes or dynamic backgrounds. For extra credit, research and implement a more advanced background subtraction technique in the file `solution/advanced_detector.py`.

*   **Task:** Implement the function `advanced_motion_detector(video_path)`, which takes a path to a video file and produces an output video showing the detected motion that does better than the simple frame differencing method. You are highly encouraged to use an LLM for research and to help generate the initial code for this.
*   **Deliverables:**
    1.  Your implementation in `solution/advanced_detector.py`.
    2.  A new section in `REPORT.md` titled "Extra Credit: Advanced Motion Detection".
    3.  In this section, you must include:
        *   **The Prompt:** The exact prompt(s) you used with an LLM that helped you solve this.
        *   **The Result:** A side-by-side comparison (GIFs or still images) showing the output of your simple detector vs. your advanced detector on the `failure_case` data. Save your output GIF in the `output/` folder.
        *   **Analysis:** A paragraph explaining *why* your method is more robust than simple frame differencing for this specific failure case.

## 6. Deliverables Summary

-   **Code:**
    -   Completed `solution/motion_detector.py`.
    -   (Optional) Completed `solution/advanced_detector.py`.
-   **Report:**
    -   A `REPORT.md` file in the root directory containing your analysis for Part 2 and (optionally) Part 3.
-   **Output Files:**
    -   The output image from the failure case analysis (`output/failure_case_output.png`).
    -   (Optional) The output GIF from your advanced detector (`output/advanced_detector_output.gif`).
