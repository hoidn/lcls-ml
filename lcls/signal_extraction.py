import matplotlib.pyplot as plt
import numpy as np

def filter_and_sum_histograms(histograms, energies, Emin, Emax):
    # Create a mask based on energy constraints
    energy_mask = (energies >= Emin) & (energies <= Emax)

    # Apply the energy mask to the histograms
    filtered_histograms = histograms[energy_mask, :, :]

    # Sum along the 0th axis
    summed_histograms = np.sum(filtered_histograms, axis=0)

    return summed_histograms

def calculate_signal_background_noI0(data, infilled_clusters, buf1=10, buf2=20):
    # Run the analysis
#     emd_values, p_values, _, roi_connected_cluster, _, _ = run_histogram_analysis(
#         data, bin_boundaries, hist_start_bin, roi_x_start, roi_x_end, roi_y_start, roi_y_end,
#         threshold=threshold)
    local_histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)
#     filtered_clusters = filter_negative_clusters_by_size(roi_connected_cluster, M=50)
    integrated_counts = filter_and_sum_histograms(local_histograms, energies, 8, 10)

    buffer = create_continuous_buffer(infilled_clusters, buf1)
    buffer = create_continuous_buffer(infilled_clusters | buffer, buf2)
    signal, bg = background_subtraction(integrated_counts, infilled_clusters, buffer)
    #mark
    # Poisson statistics for variance
    var_signal = signal  # variance for signal
    var_bg = bg  # variance for background

    # Combined variance in quadrature
    total_var = np.sqrt(var_signal**2 + var_bg**2)

    return signal, bg, total_var

def calculate_signal_noI0(data, infilled_clusters, buf1=20, buf2=20):
    integrated_counts = filter_and_sum_histograms(histograms, energies, 8, 10)

    # Poisson statistics for variance
    var_signal = signal  # variance for signal

    return signal, var_signal

import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from typing import Union

def create_continuous_buffer(infilled_clusters: np.ndarray, thickness: int = 10, num_pixels: int = None) -> np.ndarray:
    buffer = np.zeros_like(infilled_clusters, dtype=bool)
    dilated_region = binary_dilation(infilled_clusters, iterations=thickness)
    buffer = dilated_region & (~infilled_clusters)
    if num_pixels is not None:
        buffer_pixels = np.argwhere(buffer)
        if buffer_pixels.shape[0] > num_pixels:
            remove_indices = np.random.choice(buffer_pixels.shape[0], size=buffer_pixels.shape[0] - num_pixels, replace=False)
            buffer_pixels_to_remove = buffer_pixels[remove_indices]
            buffer[buffer_pixels_to_remove[:, 0], buffer_pixels_to_remove[:, 1]] = False
    return buffer

def calculate_average_counts(integrated_counts: np.ndarray, buffer: np.ndarray) -> Union[float, None]:
    counts_in_buffer = integrated_counts[buffer]
    if counts_in_buffer.size == 0:
        return None
    M = np.mean(counts_in_buffer)
    return M

def calculate_total_counts(integrated_counts: np.ndarray, infilled_clusters: np.ndarray) -> int:
    counts_in_signal = integrated_counts[infilled_clusters]
    S = np.sum(counts_in_signal)
    return S

def background_subtraction(integrated_counts: np.ndarray, infilled_clusters: np.ndarray, buffer: np.ndarray) -> Union[float, None]:
    # TODO unequal signal and background
    N = np.sum(infilled_clusters)
    M = calculate_total_counts(integrated_counts, buffer)
    if M is None:
        return None
    S = calculate_total_counts(integrated_counts, infilled_clusters)
    return S, M * N / np.sum(buffer)
    result = S - N * M
    return result
