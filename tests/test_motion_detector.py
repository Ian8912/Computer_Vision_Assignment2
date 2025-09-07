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
from solution.motion_detector import generate_motion_mask, clean_and_find_bbox

# --- Helper data and paths ---
TEST_FRAME_PATH = 'data/walkstraight/frame0062.tif'
NOISY_MASK_PATH = 'tests/data/noisy_mask.npy'


# --- Tests for Part 1, Function 1: generate_motion_mask ---

@pytest.mark.timeout(15)
def test_generate_mask_runs_and_returns_correct_type():
    """[10 Points] generate_motion_mask runs without errors and returns a NumPy array."""
    try:
        mask = generate_motion_mask(TEST_FRAME_PATH, k=10, threshold=10)
        assert isinstance(mask, np.ndarray), "Function should return a NumPy array."
        assert mask.dtype == np.uint8, "Mask should be of type uint8."
    except Exception as e:
        pytest.fail(f"generate_motion_mask failed to run: {e}")

@pytest.mark.timeout(15)
def test_generate_mask_dimensions():
    """[10 Points] generate_motion_mask returns a mask with the correct dimensions."""
    original_img = cv2.imread(TEST_FRAME_PATH, cv2.IMREAD_GRAYSCALE)
    expected_shape = original_img.shape
    
    mask = generate_motion_mask(TEST_FRAME_PATH, k=10, threshold=10)
    assert mask.shape == expected_shape, \
        f"Expected mask shape {expected_shape}, but got {mask.shape}."

@pytest.mark.timeout(15)
def test_generate_mask_content():
    """[15 Points] generate_motion_mask produces a reasonably correct motion mask."""
    mask = generate_motion_mask(TEST_FRAME_PATH, k=10, threshold=10)
    
    assert np.all(np.logical_or(mask == 0, mask == 255)), "Mask must be binary (0s and 255s)."
    
    num_white_pixels = np.sum(mask == 255)
    min_expected = 1700
    max_expected = 2000
    assert min_expected <= num_white_pixels <= max_expected, \
        f"Expected between {min_expected} and {max_expected} motion pixels, but found {num_white_pixels}. Check your differencing or thresholding logic."


# --- Tests for Part 1, Function 2: clean_and_find_bbox ---

@pytest.fixture
def noisy_mask():
    """A pytest fixture to load the pre-defined noisy mask for tests."""
    if not os.path.exists(NOISY_MASK_PATH):
        pytest.fail(f"Required test file not found: {NOISY_MASK_PATH}")
    return np.load(NOISY_MASK_PATH)

@pytest.mark.timeout(15)
def test_clean_and_find_bbox_runs_and_returns_correct_type(noisy_mask):
    """[10 Points] clean_and_find_bbox runs and returns a tuple of 4 integers."""
    try:
        bbox = clean_and_find_bbox(noisy_mask)
        assert isinstance(bbox, tuple), "Function should return a tuple."
        assert len(bbox) == 4, "Tuple should contain 4 elements."
        assert all(isinstance(val, int) for val in bbox), "All elements in the tuple should be integers."
    except Exception as e:
        pytest.fail(f"clean_and_find_bbox failed to run: {e}")

@pytest.mark.timeout(15)
def test_clean_and_find_bbox_correctness(noisy_mask):
    """[25 Points] clean_and_find_bbox correctly cleans the mask and finds the largest component's bbox."""
    student_bbox = clean_and_find_bbox(noisy_mask)
    
    # Expected bbox: (top=50, bottom=200, left=100, right=180)
    expected_bbox = (50, 200, 100, 180) 
    
    # Define a tolerance for each coordinate
    tolerance = 5 # pixels
    
    # Unpack the tuples for easier comparison
    s_top, s_bottom, s_left, s_right = student_bbox
    e_top, e_bottom, e_left, e_right = expected_bbox
    
    # Assert each coordinate is within the tolerance
    assert abs(s_top - e_top) <= tolerance, f"Top coordinate is out of tolerance. Expected around {e_top}, got {s_top}."
    assert abs(s_bottom - e_bottom) <= tolerance, f"Bottom coordinate is out of tolerance. Expected around {e_bottom}, got {s_bottom}."
    assert abs(s_left - e_left) <= tolerance, f"Left coordinate is out of tolerance. Expected around {e_left}, got {s_left}."
    assert abs(s_right - e_right) <= tolerance, f"Right coordinate is out of tolerance. Expected around {e_right}, got {s_right}."
    
    # Optional: Also check that the overall size is similar
    student_height = s_bottom - s_top
    student_width = s_right - s_left
    expected_height = e_bottom - e_top
    expected_width = e_right - e_left
    
    assert abs(student_height - expected_height) <= tolerance * 2, "Bounding box height is significantly different from expected."
    assert abs(student_width - expected_width) <= tolerance * 2, "Bounding box width is significantly different from expected."


@pytest.mark.timeout(15)
def test_clean_and_find_bbox_no_components():
    """[Bonus 5 Points - defensive programming] clean_and_find_bbox handles masks with no components."""
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    empty_mask[10, 10] = 255
    
    bbox = clean_and_find_bbox(empty_mask)
    expected_bbox = (0, 0, 0, 0)
    
    assert bbox == expected_bbox, \
        f"Expected {expected_bbox} for an empty/noisy mask, but got {bbox}. Ensure your function handles cases with no valid components found after cleaning."