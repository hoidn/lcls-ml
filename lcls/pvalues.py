#from . import histogram_analysis as hist
#from .histogram_analysis import run_histogram_analysis

import histogram_analysis as hist
import numpy as np
run_histogram_analysis = hist.run_histogram_analysis
calculate_histograms = hist.calculate_histograms
create_continuous_buffer = hist.create_continuous_buffer
create_background_mask = hist.create_background_mask

def perform_permutation_test(sample1, sample2, num_permutations, random_state=None):
    # Handle empty arrays or arrays with NaNs
    if len(sample1) == 0 or len(sample2) == 0 or np.isnan(sample1).any() or np.isnan(sample2).any():
        return np.nan

    np.random.seed(random_state)

    # Use absolute value for observed_diff
    observed_diff = np.abs(np.mean(sample1) - np.mean(sample2))
    all_samples = np.concatenate([sample1, sample2])
    count_extreme_values = 0

    for _ in range(num_permutations):
        np.random.shuffle(all_samples)
        new_diff = np.abs(np.mean(all_samples[:len(sample1)]) - np.mean(all_samples[len(sample1):]))
        if new_diff >= observed_diff:
            count_extreme_values += 1

    p_value = count_extreme_values / num_permutations
    return p_value

# Wrapper function to calculate EMD values with a specified ROI
def calculate_emd_values_with_roi(histograms, bin_boundaries, hist_start_bin, roi_coordinates):
    roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_coordinates
    average_histogram = hist.get_average_roi_histogram(histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end)
    emd_values = hist.calculate_emd_values(histograms, average_histogram)
    return emd_values, average_histogram

# Helper function to get background EMD values based on signal mask
def get_background_emd_values(signal_mask, z_offset):
    # Shift the mask up by 'z_offset' indices in the 0th mask dimension
    shifted_mask = np.roll(signal_mask, -z_offset, axis=0)
    return shifted_mask

def compute_aggregate_pvals(bin_boundaries, hist_start_bin, roi_coordinates, threshold,
                            num_permutations, z_offset,
                            histograms=None, data=None,
                            signal_mask=None,
                            background_mask=None,
                            random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    if data is None:
        assert histograms is not None

    # Check if histograms is None and run_histogram_analysis has not been called
    if signal_mask is None:
        analysis_results = run_histogram_analysis(data = data, histograms = histograms,
                                                bin_boundaries = bin_boundaries, hist_start_bin = hist_start_bin,
                                                roi_x_start = roi_coordinates[0], roi_x_end = roi_coordinates[1],
                                                roi_y_start = roi_coordinates[2], roi_y_end = roi_coordinates[3],
                                                threshold=threshold)
        signal_mask = analysis_results['signal_mask']
    if histograms is None:
        histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)
        #histograms = analysis_results['histograms']

    emd_values, average_histogram = calculate_emd_values_with_roi(histograms, bin_boundaries, hist_start_bin, roi_coordinates)
    signal_emd_values = emd_values[signal_mask].flatten()

    roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_coordinates
    if background_mask is None:
        background_mask = get_background_emd_values(signal_mask, z_offset)

    background_emd_values = emd_values[background_mask].flatten()

    p_value = perform_permutation_test(signal_emd_values, background_emd_values,
                                       num_permutations, random_state)
    return p_value

def compute_aggregate_pvals_with_custom_background(bin_boundaries, hist_start_bin,
                                                   roi_coordinates,
                                                   background_mask_multiple, num_permutations=1000,
                                                   z_offset=20, thickness=10,
                                                   random_state=None, signal_mask=None,
                                                   threshold=None, data=None, histograms=None):
    """
    Computes aggregate p-values using a custom background mask, also returning the background mask.
    If a signal mask is provided, it is used directly instead of calculating it based on threshold.
    If 'data' is provided without 'histograms', it is passed to 'run_histogram_analysis'.
    If both 'data' and 'histograms' are provided, the function raises an error.

    Parameters:
    - bin_boundaries: The boundaries of the bins in the histogram.
    - hist_start_bin: The starting bin for the histogram analysis.
    - roi_coordinates: A tuple (roi_x_start, roi_x_end, roi_y_start, roi_y_end) defining the ROI.
    - background_mask_multiple: Multiple of the number of pixels in the signal mask to approximate in the background mask.
    - num_permutations: Number of permutations for statistical testing (default=100).
    - z_offset: Offset value for z-axis in the analysis (default=20).
    - thickness: Thickness of the background buffer (default=10).
    - random_state: Random state for reproducibility (default=None).
    - signal_mask: An optional boolean array representing the signal mask (default=None).
    - threshold: An optional threshold value for the histogram analysis; should not be provided if signal_mask is given (default=None).
    - data: Optional data array to be passed to 'run_histogram_analysis' instead of 'histograms'.
    - histograms: The histogram data, should not be provided if 'data' is given.

    Returns:
    - A dictionary containing the background mask, the analysis result, and the aggregate p-values.
    """

    if histograms is not None and data is not None:
        raise ValueError("Either 'histograms' or 'data' should be provided, not both.")

    if signal_mask is not None and threshold is not None:
        raise ValueError("If signal_mask is provided, threshold should not be provided.")

    # Calculate the signal mask if not provided
    if signal_mask is None:
        analysis_result = run_histogram_analysis(
            bin_boundaries=bin_boundaries, hist_start_bin=hist_start_bin,
            roi_x_start=roi_coordinates[0], roi_x_end=roi_coordinates[1],
            roi_y_start=roi_coordinates[2], roi_y_end=roi_coordinates[3],
            data=data, histograms=histograms, threshold=threshold
        )
        signal_mask = analysis_result['signal_mask']
    else:
        analysis_result = None  # No analysis result if signal_mask is provided

    # Calculate the approximate number of pixels for the background mask
    #num_pixels_signal_mask = np.sum(signal_mask)
    #num_pixels_background_mask = int(num_pixels_signal_mask * background_mask_multiple)

    # Create the background mask using the new function
    background_mask = create_background_mask(signal_mask, background_mask_multiple, thickness)

    # Compute aggregate p-values using the custom background mask
    p_value = compute_aggregate_pvals(
        bin_boundaries, hist_start_bin, roi_coordinates, threshold,
        num_permutations, z_offset, histograms=histograms, data = data,
        signal_mask=signal_mask, background_mask=background_mask, random_state=random_state
    )

    # Returning results in a dictionary
    return {
        "background_mask": background_mask,
        "signal_mask": signal_mask,
        "aggregate_p_value": p_value
    }
