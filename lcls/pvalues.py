from . import histogram_analysis as hist
import numpy as np

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

# # Top-level function to execute a permutation test to compare EMD values between a signal ROI and background
# def compute_aggregate_pvals(bin_boundaries, hist_start_bin, roi_coordinates, threshold,
#                                             num_permutations, z_offset,
#                                             histograms = None, data = None,
#                                             signal_mask = None,
#                                             background_mask = None,
#                                             random_state=None):
#     if random_state is not None:
#         np.random.seed(random_state)
#     if histograms is None:
#         histograms = hist.calculate_histograms(data, bin_boundaries, hist_start_bin)
#     if signal_mask is None:
#         _, _, _, _, _, signal_mask = hist.run_histogram_analysis(
#             data, bin_boundaries, hist_start_bin, roi_coordinates[0], roi_coordinates[1],
#             roi_coordinates[2], roi_coordinates[3], num_permutations=num_permutations, threshold=threshold)

#     emd_values, average_histogram = calculate_emd_values_with_roi(histograms, bin_boundaries, hist_start_bin, roi_coordinates)
#     signal_emd_values = emd_values[signal_mask].flatten()

#     roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_coordinates
#     if background_mask is None:
#         background_mask = get_background_emd_values(signal_mask, z_offset)

#     background_emd_values = emd_values[background_mask].flatten()

#     p_value = perform_permutation_test(signal_emd_values, background_emd_values,
#                                        num_permutations, random_state)
#     return p_value

from .histogram_analysis import run_histogram_analysis
def compute_aggregate_pvals(bin_boundaries, hist_start_bin, roi_coordinates, threshold,
                            num_permutations, z_offset,
                            histograms=None, data=None,
                            signal_mask=None,
                            background_mask=None,
                            random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Check if histograms is None and run_histogram_analysis has not been called
    if histograms is None and data is not None:
        analysis_results = run_histogram_analysis(data = data, histograms = histograms,
                                                  bin_boundaries = bin_boundaries, hist_start_bin = hist_start_bin,
                                                  roi_x_start = roi_coordinates[0], roi_x_end = roi_coordinates[1],
                                                  roi_y_start = roi_coordinates[2], roi_y_end = roi_coordinates[3],
                                                  num_permutations=num_permutations,
                                                  threshold=threshold)
        histograms = analysis_results['histograms']
        signal_mask = analysis_results['signal_mask']

    emd_values, average_histogram = calculate_emd_values_with_roi(histograms, bin_boundaries, hist_start_bin, roi_coordinates)
    signal_emd_values = emd_values[signal_mask].flatten()

    roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_coordinates
    if background_mask is None:
        background_mask = get_background_emd_values(signal_mask, z_offset)

    background_emd_values = emd_values[background_mask].flatten()

    p_value = perform_permutation_test(signal_emd_values, background_emd_values,
                                       num_permutations, random_state)
    return p_value
