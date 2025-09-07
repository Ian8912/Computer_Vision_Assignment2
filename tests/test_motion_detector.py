# tests/test_motion_detector.py

import os
import sys

# Ensure the repository root is importable so `solution` can be imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
import pytest
import os

# Pytest will automatically discover the 'solution' directory as a package
# because of the __init__.py file and the project structure.
# No sys.path manipulation is needed.
from solution.motion_detector import generate_motion_mask, clean_and_find_bbox

# --- Helper data and paths ---
# We use a known frame index for consistent testing
TEST_VIDEO_PATH = 'data/walkstraight.mp4'
TEST_FRAME_INDEX = 52
NOISY_MASK_PATH = 'tests/data/noisy_mask.npy'


def _read_gray_video_frame(video_path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if index < 0:
        index = 0
    if total > 0 and index >= total:
        index = total - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# --- Tests for Part 1, Function 1: generate_motion_mask ---

@pytest.mark.timeout(15)
def test_generate_mask_runs_and_returns_correct_type():
    """[6 Points] generate_motion_mask runs without errors and returns a NumPy array."""
    try:
        # Call the function with all required parameters
        mask = generate_motion_mask(TEST_VIDEO_PATH, t=TEST_FRAME_INDEX, k=10, threshold=10, blur_ksize=5)
        assert isinstance(mask, np.ndarray), "Function should return a NumPy array."
        assert mask.dtype == np.uint8, "Mask should be of type uint8."
    except Exception as e:
        pytest.fail(f"generate_motion_mask failed to run: {e}")

@pytest.mark.timeout(15)
def test_generate_mask_dimensions():
    """[6 Points] generate_motion_mask returns a mask with the correct dimensions."""
    original_img = _read_gray_video_frame(TEST_VIDEO_PATH, TEST_FRAME_INDEX)
    expected_shape = original_img.shape
    
    mask = generate_motion_mask(TEST_VIDEO_PATH, t=TEST_FRAME_INDEX, k=10, threshold=10, blur_ksize=5)
    assert mask.shape == expected_shape, \
        f"Expected mask shape {expected_shape}, but got {mask.shape}."

@pytest.mark.timeout(15)
def test_generate_mask_content():
    """[12 Points] generate_motion_mask produces a reasonably correct motion mask."""
    mask = generate_motion_mask(TEST_VIDEO_PATH, t=TEST_FRAME_INDEX, k=10, threshold=10, blur_ksize=5)
    
    assert np.all(np.logical_or(mask == 0, mask == 255)), "Mask must be binary (0s and 255s)."

    # Save mask for debugging purposes (comment out in production)
    cv2.imwrite("output/motion_mask.png", mask)

    # The expected range of white pixels is now smaller and tighter because the
    # pre-processing blur removes a significant amount of sensor noise.
    num_white_pixels = np.sum(mask == 255)
    min_expected = 4500
    max_expected = 4800
    assert min_expected <= num_white_pixels <= max_expected, \
        f"Expected between {min_expected} and {max_expected} motion pixels, but found {num_white_pixels}. Check your blurring, differencing, or thresholding logic."


# --- Tests for Part 1, Function 2: clean_and_find_bbox ---

@pytest.fixture
def noisy_mask():
    """A pytest fixture to load the pre-defined noisy mask for tests."""
    if not os.path.exists(NOISY_MASK_PATH):
        pytest.fail(f"Required test file not found: {NOISY_MASK_PATH}")
    return np.load(NOISY_MASK_PATH)

@pytest.mark.timeout(15)
def test_clean_and_find_bbox_runs_and_returns_correct_types(noisy_mask):
    """[6 Points] clean_and_find_bbox runs and returns correct types (tuple, ndarray)."""
    try:
        result = clean_and_find_bbox(noisy_mask)
        assert isinstance(result, tuple) and len(result) == 2, "Function should return a tuple of two items."
        
        bbox, cleaned_mask_with_bbox = result
        assert isinstance(bbox, tuple) and len(bbox) == 4, "First item should be a bounding box tuple of 4 elements."
        assert all(isinstance(val, int) for val in bbox), "All elements in the bbox tuple should be integers."
        assert isinstance(cleaned_mask_with_bbox, np.ndarray) and cleaned_mask_with_bbox.dtype == np.uint8, \
            "Second item should be a uint8 NumPy array (the cleaned mask)."

    except Exception as e:
        pytest.fail(f"clean_and_find_bbox failed to run: {e}")

@pytest.mark.timeout(15)
def test_clean_and_find_bbox_correctness(noisy_mask):
    """[20 Points] clean_and_find_bbox correctly cleans the mask and finds the largest component's bbox."""
    student_bbox, cleaned_mask_with_bbox = clean_and_find_bbox(noisy_mask)
    assert isinstance(cleaned_mask_with_bbox, np.ndarray), "Final mask should be a NumPy array."
    assert cleaned_mask_with_bbox.dtype == np.uint8, "Final mask should be of type uint8."
    assert np.all(np.logical_or(cleaned_mask_with_bbox == 0, cleaned_mask_with_bbox == 255)), "Final mask must be binary (0s and 255s)."

    # Expected bbox for the largest component in tests/data/noisy_mask.npy:
    # (top=50, bottom=200, left=100, right=180)
    expected_bbox = (50, 200, 100, 180) 
    
    # Use a tolerance to account for minor implementation differences.
    tolerance = 5 # pixels
    
    # Unpack the tuples for clearer assertion messages
    s_top, s_bottom, s_left, s_right = student_bbox
    e_top, e_bottom, e_left, e_right = expected_bbox

    # Assert each coordinate is within the tolerance
    assert abs(s_top - e_top) <= tolerance, f"Top coordinate is out of tolerance. Expected around {e_top}, got {s_top}."
    assert abs(s_bottom - e_bottom) <= tolerance, f"Bottom coordinate is out of tolerance. Expected around {e_bottom}, got {s_bottom}."
    assert abs(s_left - e_left) <= tolerance, f"Left coordinate is out of tolerance. Expected around {e_left}, got {s_left}."
    assert abs(s_right - e_right) <= tolerance, f"Right coordinate is out of tolerance. Expected around {e_right}, got {s_right}."

@pytest.mark.timeout(15)
def test_clean_and_find_bbox_no_components():
    """[Bonus 5 Points - defensive programming] clean_and_find_bbox handles masks with no components."""
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    empty_mask[10, 10] = 255
    
    # Run the student's function
    result = clean_and_find_bbox(empty_mask)

    assert isinstance(result, tuple) and len(result) == 2, \
        "Function must always return a tuple of two items (bbox, cleaned_mask_with_bbox), even when no components are found."
    
    # Now that we know the structure is correct, we can safely unpack
    student_bbox, student_cleaned_mask_with_bbox = result
    
    # Test 1: Check the bounding box
    expected_bbox = (0, 0, 0, 0)
    assert student_bbox == expected_bbox, \
        f"Expected {expected_bbox} for an empty/noisy mask, but got {student_bbox}."
        
    # Test 2: Check the cleaned mask
    # The opening should have removed the single noise pixel, resulting in a black image.
    assert np.sum(student_cleaned_mask_with_bbox) == 0, \
        "The cleaned mask should be completely black when the input contains only small noise."


@pytest.mark.timeout(15)
def test_clean_and_find_bbox_with_generated_mask():
    """[10 Points] clean_and_find_bbox works when given a mask produced by generate_motion_mask."""
    mask = generate_motion_mask(TEST_VIDEO_PATH, t=TEST_FRAME_INDEX, k=10, threshold=10, blur_ksize=5)

    result = clean_and_find_bbox(mask)
    assert isinstance(result, tuple) and len(result) == 2, "Function should return (bbox, cleaned_mask_with_bbox)."

    bbox, cleaned_mask_with_bbox = result
    assert isinstance(bbox, tuple) and len(bbox) == 4, "First item should be a 4-tuple bbox."
    assert isinstance(cleaned_mask_with_bbox, np.ndarray) and cleaned_mask_with_bbox.dtype == np.uint8, \
        "Second item should be a uint8 NumPy array."

    # Basic sanity checks
    h, w = mask.shape
    top, bottom, left, right = bbox
    assert 0 <= top <= bottom <= h, "BBox rows must be within image bounds."
    assert 0 <= left <= right <= w, "BBox cols must be within image bounds."
    assert cleaned_mask_with_bbox.shape == mask.shape, "Output mask shape must match input mask shape."
    assert np.all(np.logical_or(cleaned_mask_with_bbox == 0, cleaned_mask_with_bbox == 255)), "Mask must be binary."

    # If a non-empty bbox is returned, verify that a rectangle outline exists at the bbox edges
    if bbox != (0, 0, 0, 0):
        # Our drawing uses right-1 and bottom-1 as inclusive endpoints
        right_inclusive = max(left, right - 1)
        bottom_inclusive = max(top, bottom - 1)

        # Check that there are white pixels along each rectangle edge
        assert np.any(cleaned_mask_with_bbox[top, left:right_inclusive + 1] == 255), "Top edge of bbox not drawn."
        assert np.any(cleaned_mask_with_bbox[bottom_inclusive, left:right_inclusive + 1] == 255), "Bottom edge of bbox not drawn."
        assert np.any(cleaned_mask_with_bbox[top:bottom_inclusive + 1, left] == 255), "Left edge of bbox not drawn."
        assert np.any(cleaned_mask_with_bbox[top:bottom_inclusive + 1, right_inclusive] == 255), "Right edge of bbox not drawn."

        # Save the cleaned mask with the expected bbox
        cv2.imwrite("output/cleaned_mask_with_bbox.png", cleaned_mask_with_bbox)


@pytest.mark.timeout(20)
def test_bbox_frame_52():
    mask = generate_motion_mask(TEST_VIDEO_PATH, t=52, k=10, threshold=10, blur_ksize=5)
    bbox, cleaned_mask_with_bbox = clean_and_find_bbox(mask)

    assert isinstance(cleaned_mask_with_bbox, np.ndarray)
    assert cleaned_mask_with_bbox.dtype == np.uint8

    s_top, s_bottom, s_left, s_right = bbox
    e_top, e_bottom, e_left, e_right = (63, 219, 184, 244)

    tolerance = 5
    assert abs(s_top - e_top) <= tolerance, f"Top out of tolerance for frame 52: expected {e_top}, got {s_top}"
    assert abs(s_bottom - e_bottom) <= tolerance, f"Bottom out of tolerance for frame 52: expected {e_bottom}, got {s_bottom}"
    assert abs(s_left - e_left) <= tolerance, f"Left out of tolerance for frame 52: expected {e_left}, got {s_left}"
    assert abs(s_right - e_right) <= tolerance, f"Right out of tolerance for frame 52: expected {e_right}, got {s_right}"

@pytest.mark.timeout(20)
def test_bbox_frame_62():
    mask = generate_motion_mask(TEST_VIDEO_PATH, t=62, k=10, threshold=10, blur_ksize=5)
    bbox, cleaned_mask_with_bbox = clean_and_find_bbox(mask)

    assert isinstance(cleaned_mask_with_bbox, np.ndarray)
    assert cleaned_mask_with_bbox.dtype == np.uint8

    s_top, s_bottom, s_left, s_right = bbox
    e_top, e_bottom, e_left, e_right = (59, 218, 124, 196)

    tolerance = 5
    assert abs(s_top - e_top) <= tolerance, f"Top out of tolerance for frame 62: expected {e_top}, got {s_top}"
    assert abs(s_bottom - e_bottom) <= tolerance, f"Bottom out of tolerance for frame 62: expected {e_bottom}, got {s_bottom}"
    assert abs(s_left - e_left) <= tolerance, f"Left out of tolerance for frame 62: expected {e_left}, got {s_left}"
    assert abs(s_right - e_right) <= tolerance, f"Right out of tolerance for frame 62: expected {e_right}, got {s_right}"

@pytest.mark.timeout(20)
def test_bbox_frame_83():
    mask = generate_motion_mask(TEST_VIDEO_PATH, t=83, k=10, threshold=10, blur_ksize=5)
    bbox, cleaned_mask_with_bbox = clean_and_find_bbox(mask)

    assert isinstance(cleaned_mask_with_bbox, np.ndarray)
    assert cleaned_mask_with_bbox.dtype == np.uint8

    s_top, s_bottom, s_left, s_right = bbox
    e_top, e_bottom, e_left, e_right = (58, 212, 41, 137)

    tolerance = 5
    assert abs(s_top - e_top) <= tolerance, f"Top out of tolerance for frame 83: expected {e_top}, got {s_top}"
    assert abs(s_bottom - e_bottom) <= tolerance, f"Bottom out of tolerance for frame 83: expected {e_bottom}, got {s_bottom}"
    assert abs(s_left - e_left) <= tolerance, f"Left out of tolerance for frame 83: expected {e_left}, got {s_left}"
    assert abs(s_right - e_right) <= tolerance, f"Right out of tolerance for frame 83: expected {e_right}, got {s_right}"

@pytest.mark.timeout(20)
def test_bbox_frame_95():
    mask = generate_motion_mask(TEST_VIDEO_PATH, t=95, k=10, threshold=10, blur_ksize=5)
    bbox, cleaned_mask_with_bbox = clean_and_find_bbox(mask)

    assert isinstance(cleaned_mask_with_bbox, np.ndarray)
    assert cleaned_mask_with_bbox.dtype == np.uint8

    s_top, s_bottom, s_left, s_right = bbox
    e_top, e_bottom, e_left, e_right = (54, 210, 19, 69)

    tolerance = 5
    assert abs(s_top - e_top) <= tolerance, f"Top out of tolerance for frame 95: expected {e_top}, got {s_top}"
    assert abs(s_bottom - e_bottom) <= tolerance, f"Bottom out of tolerance for frame 95: expected {e_bottom}, got {s_bottom}"
    assert abs(s_left - e_left) <= tolerance, f"Left out of tolerance for frame 95: expected {e_left}, got {s_left}"
    assert abs(s_right - e_right) <= tolerance, f"Right out of tolerance for frame 95: expected {e_right}, got {s_right}"