import numpy as np
from scipy.ndimage import distance_transform_edt


def generate_mask(shape, roi):
    """
    Generate a 2D mask with ones in the specified ROI and zeros elsewhere.

    Parameters:
    - shape: tuple, the shape of the mask (height, width).
    - roi: tuple, the region of interest in the format (row_start, row_end, col_start, col_end).

    Returns:
    - mask: 2D numpy array, generated mask.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    row_start, row_end, col_start, col_end = roi
    mask[row_start:row_end, col_start:col_end] = 1
    return mask.astype(bool)

from scipy.ndimage import binary_erosion, generate_binary_structure
def erode_signal_mask(signal_mask, iterations=1, structure=None):
    """
    Apply binary erosion to a signal mask.

    Parameters:
    - signal_mask: 2D numpy array of boolean values (True/False or 1/0).
    - iterations: int, the number of times to erode the mask.
    - structure: 2D array, structuring element used for the erosion.
                 If None, a cross-shaped structure is used by default.

    Returns:
    - eroded_signal_mask: 2D numpy array, eroded signal mask.
    """
    if structure is None:
        structure = generate_binary_structure(2, 1)  # 2D cross-shaped structure

    eroded_signal_mask = binary_erosion(signal_mask, structure=structure, iterations=iterations)
    return eroded_signal_mask


def erode_to_target(signal_mask, target_true_count, structure=None):
    """
    Erode a signal mask until it has approximately the target number of True pixels.

    Parameters:
    - signal_mask: 2D numpy array of boolean values (True/False or 1/0).
    - target_true_count: int, target number of True pixels in the eroded mask.
    - structure: 2D array, structuring element used for the erosion.
                 If None, a cross-shaped structure is used by default.

    Returns:
    - eroded_signal_mask: 2D numpy array, eroded signal mask.
    """
    if structure is None:
        structure = generate_binary_structure(2, 1)

    current_mask = signal_mask.copy()
    while np.sum(current_mask) > target_true_count:
        current_mask = binary_erosion(current_mask, structure)
        if np.sum(current_mask) <= target_true_count:
            break

    return current_mask


def set_nearest_neighbors(histograms, mask, roi_crop):
    # Extract the ROI from the mask
    roi_mask = mask[roi_crop[0]:roi_crop[1], roi_crop[2]:roi_crop[3]]

    # Invert the mask for distance transform
    inverted_mask = 1 - roi_mask

    # Compute the distance transform
    distance, (indices_y, indices_x) = distance_transform_edt(inverted_mask, return_indices=True)

    # Set the values in histograms based on the nearest neighbor indices
    zero_positions = np.where(roi_mask == 0)
    for y, x in zip(*zero_positions):
        nearest_y, nearest_x = indices_y[y, x], indices_x[y, x]
        # Copying the entire slice along the 0th dimension
        histograms[:, roi_crop[0] + y, roi_crop[2] + x] = histograms[:, roi_crop[0] + nearest_y, roi_crop[2] + nearest_x]


