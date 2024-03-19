import numpy as np
from scipy.ndimage import binary_dilation
from histogram_analysis import run_histogram_analysis

def create_continuous_buffer(signal_mask: np.ndarray, initial_thickness: int = 10,
                             num_pixels: int = None, separator_thickness: int = 5) -> np.ndarray:
    """
    Create a continuous buffer around a signal mask with a gap, targeting a specific number of pixels.

    Args:
        signal_mask (np.ndarray): The original signal mask.
        initial_thickness (int): The initial thickness for dilation.
        num_pixels (int, optional): The target number of pixels in the buffer.
        separator_thickness (int): The thickness of the gap between the signal mask and the buffer.

    Returns:
        np.ndarray: The created buffer.
    """
    if num_pixels > np.prod(signal_mask.shape) - np.sum(signal_mask):
        raise ValueError
    assert signal_mask.sum() > 0
    # Create a gap between the signal mask and the buffer
    dilated_signal = binary_dilation(signal_mask, iterations=separator_thickness)

    # Adjust the buffer to meet or exceed the target number of pixels
    current_num_pixels = 0
    thickness = 0
    while num_pixels is not None and current_num_pixels < num_pixels:
        thickness += 1
        buffer = binary_dilation(dilated_signal, iterations=thickness) & (~signal_mask)
        current_num_pixels = np.sum(buffer)

    return buffer

def create_background_mask(signal_mask, background_mask_multiple, thickness, separator_thickness=5):
    num_pixels_signal_mask = np.sum(signal_mask)
    num_pixels_background_mask = int(num_pixels_signal_mask * background_mask_multiple)
    buffer = create_continuous_buffer(signal_mask,
                                      initial_thickness=thickness, num_pixels=num_pixels_background_mask,
                                      separator_thickness=separator_thickness)
    return buffer

def compute_signal_mask(bin_boundaries, hist_start_bin, roi_coordinates, threshold,
                        data=None, histograms=None):
    """
    Computes the signal mask based on the given parameters and threshold.

    :param bin_boundaries: The boundaries for histogram bins.
    :param hist_start_bin: The starting bin for the histogram.
    :param roi_coordinates: A tuple of (roi_x_start, roi_x_end, roi_y_start, roi_y_end).
    :param data: The 3D numpy array containing the data.
    :param threshold: The threshold value for the analysis.
    :return: The computed signal mask.
    """
    roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_coordinates
    res = run_histogram_analysis(histograms=histograms, bin_boundaries=bin_boundaries, hist_start_bin=hist_start_bin,
                                 roi_x_start=roi_x_start, roi_x_end=roi_x_end, roi_y_start=roi_y_start, roi_y_end=roi_y_end,
                                 data=data, threshold=threshold)

    return res['signal_mask']
