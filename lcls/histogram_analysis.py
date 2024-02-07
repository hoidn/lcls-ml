from typing import Union
import numpy as np
from numpy.random import choice
from scipy.ndimage import label

import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

import functools
import hashlib
import random

def memoize_subsampled(func):
    """Memoize a function by creating a hashable key using deterministically subsampled data."""
    cache = {}

    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        # Generate a hashable key from a deterministic subsample
        shape_str = str(data.shape)  # Convert shape to string to use it as a seed
        seed_value = int(hashlib.sha256(shape_str.encode()).hexdigest(), 16) % 10**8
        random.seed(seed_value)

        subsample_size = min(100, data.shape[0])  # Limit the subsample size to a maximum of 100
        subsample_indices = random.sample(range(data.shape[0]), subsample_size)
        subsample = data[subsample_indices]

        hashable_key = hashlib.sha256(subsample.tobytes()).hexdigest()

        # Check cache
        if hashable_key in cache:
            return cache[hashable_key]

        # Calculate the result and store it in the cache
        result = func(data, *args, **kwargs)
        cache[hashable_key] = result

        return result

    return wrapper

# TODO: test this version
import json

def memoize_general(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        composite_key_parts = []

        for arg in args:
            if isinstance(arg, np.ndarray):
                shape_str = str(arg.shape)
                seed_value = int(hashlib.sha256(shape_str.encode()).hexdigest(), 16) % 10**8
                random.seed(seed_value)

                subsample_size = min(100, arg.shape[0])
                subsample_indices = random.sample(range(arg.shape[0]), subsample_size)
                subsample = arg[subsample_indices]

                key_part = hashlib.sha256(subsample.tobytes()).hexdigest()
            else:
                key_part = hashlib.sha256(json.dumps(arg).encode()).hexdigest()

            composite_key_parts.append(key_part)

        composite_key = hashlib.sha256("".join(composite_key_parts).encode()).hexdigest()

        if composite_key in cache:
            return cache[composite_key]

        result = func(*args, **kwargs)
        cache[composite_key] = result

        return result

    return wrapper

#import functools
#import hashlib
#import json
#import numpy as np
#import random
#import sys
#
#def get_object_hash(obj):
#    """
#    Generate a hash for an object. Uses subsampling for large numpy arrays
#    and standard serialization for other types.
#    """
#    if isinstance(obj, np.ndarray) and obj.nbytes > 1000000:  # Subsampling for large arrays
#        seed_value = int(hashlib.sha256(str(obj.shape).encode()).hexdigest(), 16) % 10**8
#        random.seed(seed_value)
#
#        subsample_size = min(100, obj.size)  # Sample size of 100 or less
#        subsample_indices = random.sample(range(obj.size), subsample_size)
#        subsample = obj.flat[subsample_indices]
#
#        return hashlib.sha256(subsample.tobytes()).hexdigest()
#    else:
#        try:
#            return hashlib.sha256(json.dumps(obj, default=str).encode()).hexdigest()
#        except (TypeError, OverflowError):
#            return 'unserializable_type'
#
#def memoize_general(func):
#    cache = {}
#
#    @functools.wraps(func)
#    def wrapper(*args, **kwargs):
#        composite_key_parts = [get_object_hash(arg) for arg in args]
#        composite_key = hashlib.sha256("".join(composite_key_parts).encode()).hexdigest()
#
#        if composite_key in cache:
#            return cache[composite_key]
#
#        result = func(*args, **kwargs)
#        cache[composite_key] = result
#        return result
#
#    return wrapper

from numba import jit

# # Optimizing the function using Numba
def calculate_histograms(data, bin_boundaries, hist_start_bin):
    """Generate histograms for the data using vectorized methods."""
    bins = len(bin_boundaries) - 1
    rows, cols = data.shape[1], data.shape[2]
    hist_shape = (bins, rows, cols)

    # Reshape the data for easier computation
    reshaped_data = data.reshape(-1, rows * cols)

    # Perform digitization
    #bin_indices = np.digitize(reshaped_data, bin_boundaries) - 1
    bin_indices = np.digitize(reshaped_data, bin_boundaries)

    # Initialize histograms
    histograms = np.zeros(hist_shape, dtype=np.float64)

    # Populate histograms using bincount and sum along the zeroth axis
    for i in range(rows * cols):
        valid_indices = bin_indices[:, i] < bins  # Exclude indices that fall outside the bin range or equal to the last boundary
        histograms[:, i // cols, i % cols] = np.bincount(bin_indices[:, i][valid_indices], minlength=bins)
        # TODO efficiency
        # counts beyond max go into the first bin, otherwise they don't
        # contribute to the EMD
        histograms[hist_start_bin, i // cols, i % cols] += np.sum(reshaped_data[:, i] > bin_boundaries[-1])

    # Add small constant
    histograms += 1e-9
    normalized_histograms = histograms #/ (1e-9 + np.sum(histograms, axis=0))

    return normalized_histograms[hist_start_bin:, :, :]
#def calculate_histograms(data, bin_boundaries, hist_start_bin):
#    bins = len(bin_boundaries) - 1
#    rows, cols = data.shape[1], data.shape[2]
#    hist_shape = (bins, rows, cols)
#
#    reshaped_data = data.reshape(-1, rows * cols)
#    bin_indices = np.digitize(reshaped_data, bin_boundaries)
#    bin_indices[bin_indices > bins] = bins
#
#    histograms = np.zeros(hist_shape, dtype=np.float64)
#
#    for i in range(rows * cols):
#        valid_indices = bin_indices[:, i] <= bins
#        histograms[:, i // cols, i % cols] = np.bincount(bin_indices[:, i][valid_indices] - 1, minlength=bins)
#
#    histograms += 1e-9
#    return histograms[hist_start_bin:, :, :]
calculate_histograms = jit(nopython=True)(calculate_histograms)
calculate_histograms = memoize_subsampled(calculate_histograms)

def get_average_roi_histogram(histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    """Calculate the average histogram for the ROI."""
    roi_histograms = histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    average_roi_histogram = np.mean(roi_histograms, axis=(1, 2))
    return average_roi_histogram / np.sum(average_roi_histogram)

@jit(nopython=True)
def wasserstein_distance(u, v):
    cdf_u = np.cumsum(u)
    cdf_v = np.cumsum(v)
    return np.sum(np.abs(cdf_u - cdf_v))

@jit(nopython=True)
def calculate_emd_values(histograms, average_histogram):
    """Compute the Earth Mover's Distance for each histogram."""
    emd_values = np.zeros((histograms.shape[1], histograms.shape[2]))
    for i in range(histograms.shape[1]):
        for j in range(histograms.shape[2]):
            emd_values[i, j] = wasserstein_distance(histograms[:, i, j], average_histogram)
    return emd_values

def generate_null_distribution(histograms, average_histogram, roi_x_start, roi_x_end, roi_y_start, roi_y_end, num_permutations=1000):
    """
    generate a null distribution of Earth Mover's Distance (EMD) values using bootstrapping.
    """
    null_emd_values = []
    roi_histograms = histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end]

    # Number of bins in the histogram
    num_bins = roi_histograms.shape[0]

    # Number of x and y indices in the ROI
    num_x_indices = roi_x_end - roi_x_start
    num_y_indices = roi_y_end - roi_y_start

    for _ in range(num_permutations):
        # Vectorized resampling of x, y indices for each value of the 0th index (each bin of the histogram)
        random_x_indices = choice(range(num_x_indices), size=num_bins)
        random_y_indices = choice(range(num_y_indices), size=num_bins)

        # TODO this might not be the right distribution. How about
        # bootstrapping on the background emd values using Poisson sampling?
        bootstrap_sample_histogram = roi_histograms[np.arange(num_bins), random_x_indices, random_y_indices]

        null_emd_value = wasserstein_distance(bootstrap_sample_histogram, average_histogram)
        null_emd_values.append(null_emd_value)

    return np.array(null_emd_values)

def calculate_p_values(emd_values, null_distribution):
    """calculate p-values based on the observed EMD values and the null distribution."""
    return 1 - np.mean(emd_values[:, :, np.newaxis] >= null_distribution, axis=2)

def identify_roi_connected_cluster(p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    """Find the cluster connected to the ROI."""
    porous_pixels = p_values > threshold
    labeled_array, _ = label(porous_pixels)
    seed_x = (roi_x_start + roi_x_end) // 2
    seed_y = (roi_y_start + roi_y_end) // 2
    roi_cluster_label = labeled_array[seed_x, seed_y]
    return labeled_array, labeled_array == roi_cluster_label
    #return labeled_array == roi_cluster_label

def sum_histograms_by_mask(histogram_array, binary_mask, agg = 'mean'):
    # Apply the mask to select histograms
    masked_histograms = histogram_array[:, binary_mask]
    # Sum the selected histograms
    if agg == 'mean':
        summed_histogram = np.mean(masked_histograms, axis=1)
    elif agg == 'sum':
        summed_histogram = np.sum(masked_histograms, axis=1)
    return summed_histogram

def melt_negative_clusters(cluster_array, N=1):
    inverted_array = np.logical_not(cluster_array)
    dilated_array = binary_dilation(inverted_array, iterations=N)
    return np.logical_or(cluster_array, np.logical_not(dilated_array))

def filter_negative_clusters_by_size(cluster_array, M=10):
    inverted_array = np.logical_not(cluster_array)
    labeled_array, num_features = label(inverted_array)
    cluster_sizes = np.bincount(labeled_array.ravel())
    small_clusters = np.where(cluster_sizes < M)[0]
    small_cluster_mask = np.isin(labeled_array, small_clusters)
    return np.logical_or(cluster_array, small_cluster_mask)

# from your_module import run_histogram_analysis, filter_negative_clusters_by_size

# TODO refactor
def rectify_filter_mask(mask, data):
    imgs_sum = data.sum(axis = 0)
    if mask.sum() == 0:
        return ~mask
    mean_1 = imgs_sum[mask].mean()
    mean_0 = imgs_sum[~mask].mean()
    if mean_1 < mean_0:
        return ~mask
    else:
        return mask

def infill_binary_array(data, array):
    imgs_sum = data.sum(axis = 0)
    # Label connected components
    labeled_array, num_features = label(
        rectify_filter_mask(
            array, data))

    # Find the largest component
    largest_component = 0
    largest_size = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled_array == i)
        if size > largest_size:
            largest_size = size
            largest_component = i

    # Create new binary image
    infilled_array = (labeled_array == largest_component)

    return infilled_array

@memoize_general
def precompute_analysis_data(data, bin_boundaries, hist_start_bin, roi_x_start,
                    roi_x_end, roi_y_start, roi_y_end, M_value, threshold_values):
    """
    Precomputes the analysis outputs for different threshold values and stores them in a dictionary.
    """
    np.random.seed(1)
    precomputed_data = {}

    for threshold in threshold_values:
        _, p_values, _, roi_connected_cluster, _, signal_mask = run_histogram_analysis(
            data, bin_boundaries, hist_start_bin, roi_x_start, roi_x_end, roi_y_start, roi_y_end, threshold=threshold)

        precomputed_data[threshold] = {
            'p_threshold_mask': p_values < threshold,
            'filtered_clusters': signal_mask,
            'data': data
        }

    return precomputed_data

def visualize_clusters(cluster_array, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_array, cmap='gray', origin='lower')
    plt.title(title)
    plt.xlabel('y-axis')
    plt.ylabel('x-axis')
    plt.show()

def visualize_size_filtered_clusters(histogram_array, binary_mask):
    summed_histogram = sum_histograms_by_mask(histogram_array, binary_mask)
    visualize_clusters(binary_mask, 'Size-Filtered Negative Clusters')

def visualize_histogram_comparison(histogram_array, binary_mask, bin_boundaries, hist_start_bin, save_path = None,
                                  agg = 'mean'):
    energies = bin_boundaries[hist_start_bin + 1:]
    summed_histogram = sum_histograms_by_mask(histogram_array, binary_mask, agg = agg)
    summed_histogram_signal = sum_histograms_by_mask(histogram_array, ~binary_mask, agg = agg)
#     aggregate_histogram = np.sum(histogram_array, axis=(1, 2))
#     histogram_difference = aggregate_histogram - summed_histogram

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    axes[0].bar(energies, sum_histograms_by_mask(histogram_array, np.ones_like(binary_mask), agg = agg), color='green', alpha=0.7)
    axes[0].set_title('Aggregate Histogram with No Filtering')
    axes[0].set_xlabel('energy (keV)')
    axes[0].set_ylabel('mean frequency')

    axes[1].bar(energies, summed_histogram, color='blue', alpha=0.7)
    axes[1].set_title('Aggregate histogram of non-signal pixels')
    axes[1].set_xlabel('energy (keV)')
    axes[1].set_ylabel('mean frequency')

    axes[2].bar(energies, summed_histogram_signal, color='red', alpha=0.7)
    axes[2].set_title('Aggregate histogram of signal candidate clusters')
    axes[2].set_xlabel('energy (keV)')
    axes[2].set_ylabel('mean frequency')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300)
    else:
        plt.show()

def run_histogram_analysis(data = None, histograms=None, bin_boundaries=np.arange(-10, 30, 0.2), hist_start_bin=60,
                           roi_x_start=30, roi_x_end=80, roi_y_start=40, roi_y_end=90, num_permutations=1000,
                           threshold=.1, cluster_size_threshold=50):
    if histograms is None:
        histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)

    average_histogram = get_average_roi_histogram(histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end)
    emd_values = calculate_emd_values(histograms, average_histogram)

    null_distribution = generate_null_distribution(histograms, average_histogram, roi_x_start, roi_x_end, roi_y_start, roi_y_end,
                                                  num_permutations)
    p_values = calculate_p_values(emd_values, null_distribution)
    labeled_array, _ = identify_roi_connected_cluster(p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end)

    seed_x = (roi_x_start + roi_x_end) // 2
    seed_y = (roi_y_start + roi_y_end) // 2
    roi_cluster_label = labeled_array[seed_x, seed_y]
    roi_connected_cluster = (labeled_array == roi_cluster_label)

    signal_mask = filter_negative_clusters_by_size(
        rectify_filter_mask(
            roi_connected_cluster, histograms), M=cluster_size_threshold)

    # TODO refactor, this should go into run_histogram_analysis
    # Also run_histogram_analysis should return a dict
    signal_mask = infill_binary_array(histograms,
                                            signal_mask)
    return {
        "histograms": histograms,
        "average_histogram": average_histogram,
        "emd_values": emd_values,
        "null_distribution": null_distribution,
        "p_values": p_values,
        "labeled_array": labeled_array,
        "signal_mask": signal_mask
    }
#     return emd_values, p_values, labeled_array, roi_connected_cluster, null_distribution, signal_mask

def visualize_roi_connected_cluster(labeled_array, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    seed_x = (roi_x_start + roi_x_end) // 2
    seed_y = (roi_y_start + roi_y_end) // 2
    roi_cluster_label = labeled_array[seed_x, seed_y]
    roi_connected_cluster = (labeled_array == roi_cluster_label)

    visualize_clusters(roi_connected_cluster, 'Cluster Connected to the ROI')

#def filter_and_sum_histograms(histograms, energies, Emin, Emax):
#    # Create a mask based on energy constraints
#    energy_mask = (energies >= Emin) & (energies <= Emax)
#
#    # Apply the energy mask to the histograms
#    filtered_histograms = histograms[energy_mask, :, :]
#
#    # Sum along the 0th axis
#    summed_histograms = np.sum(filtered_histograms, axis=0)
#
#    return summed_histograms

def filter_and_sum_histograms(histograms, energies, Emin, Emax):
    # Calculate central energy and window size
    E0 = (Emax + Emin) / 2
    window_size = Emax - Emin

    # Initialize a mask with False values
    energy_mask = np.zeros_like(energies, dtype=bool)

    # Loop through the first three harmonics
    for n in range(1, 4):
        # Calculate window size and center energy for the nth harmonic
        harmonic_window_size = np.sqrt(n) * window_size
        harmonic_center_energy = n * E0

        # Calculate the energy bounds for the nth harmonic
        harmonic_Emin = harmonic_center_energy - harmonic_window_size / 2
        harmonic_Emax = harmonic_center_energy + harmonic_window_size / 2

        # Update the mask to include the nth harmonic energies
        energy_mask |= ((energies >= harmonic_Emin) & (energies <= harmonic_Emax))

    # Apply the energy mask to the histograms
    filtered_histograms = histograms[energy_mask, :, :]

    # Sum along the 0th axis
    summed_histograms = np.sum(filtered_histograms, axis=0)

    return summed_histograms

# TODO parameterize
def calculate_signal_background_from_histograms(local_histograms, signal_mask,
        bin_boundaries, hist_start_bin,
        background_mask_multiple=1.0, thickness=10, Emin = 8, Emax = 10):
    """
    """
    energies = bin_boundaries[hist_start_bin + 1:]
    integrated_counts = filter_and_sum_histograms(local_histograms, energies, Emin, Emax)

    # Using the new function to create the background mask
    background_mask = create_background_mask(signal_mask, background_mask_multiple, thickness)

    signal, bg = background_subtraction(integrated_counts, signal_mask, background_mask)
    # Poisson statistics for variance
    var_signal = signal  # variance for signal
    var_bg = bg  # variance for background

    nsignal = np.sum(signal_mask)
    nbg = np.sum(background_mask)

    # Combined variance in quadrature
    total_var = var_signal + (var_bg * ((nsignal / nbg)**2))

    return signal, bg, total_var

def calculate_signal_background_noI0(data, signal_mask, bin_boundaries, hist_start_bin, background_mask_multiple=1.0, thickness=10,
                                     **kwargs):
    """
    Updated version of calculate_signal_background_noI0 function to use the new background mask calculation method.

    Additional Parameters:
    - background_mask_multiple: float, multiple of the number of pixels in the signal mask for the background mask.
    - thickness: int, thickness of the background buffer.
    """
    local_histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)
    return calculate_signal_background_from_histograms(local_histograms,
                signal_mask, bin_boundaries, hist_start_bin,
                background_mask_multiple=1.0, thickness=10, **kwargs)

def calculate_average_counts(integrated_counts: np.ndarray, buffer: np.ndarray) -> Union[float, None]:
    counts_in_buffer = integrated_counts[buffer]
    if counts_in_buffer.size == 0:
        return None
    M = np.mean(counts_in_buffer)
    return M

def calculate_total_counts(integrated_counts: np.ndarray, signal_mask: np.ndarray) -> int:
    counts_in_signal = integrated_counts[signal_mask]
    S = np.sum(counts_in_signal)
    return S

def background_subtraction(integrated_counts: np.ndarray, signal_mask: np.ndarray, buffer: np.ndarray) -> Union[float, None]:
    # TODO unequal signal and background
    N = np.sum(signal_mask)
    M = calculate_total_counts(integrated_counts, buffer)
    if M is None:
        return None
    S = calculate_total_counts(integrated_counts, signal_mask)
    return S, M * N / np.sum(buffer)
    result = S - N * M
    return result

def create_background_mask(signal_mask, background_mask_multiple, thickness, separator_thickness = 5):
    """
    Creates a background mask based on the given signal mask, a multiple of its size, and thickness.

    Parameters:
    - signal_mask: numpy.ndarray, a boolean array representing the signal mask.
    - background_mask_multiple: float, multiple of the number of pixels in the signal mask for the background mask.
    - thickness: int, thickness of the background buffer.

    Returns:
    - numpy.ndarray, the calculated background mask.
    """
    num_pixels_signal_mask = np.sum(signal_mask)
    num_pixels_background_mask = int(num_pixels_signal_mask * background_mask_multiple)

    # Create the background mask using continuous buffer
    background_mask = create_continuous_buffer(signal_mask,
                initial_thickness=thickness, num_pixels=num_pixels_background_mask,
                                               separator_thickness = separator_thickness)
    return background_mask

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
    #gap_mask = dilated_signal & (~signal_mask)

    # Adjust the buffer to meet or exceed the target number of pixels
    current_num_pixels = 0
    thickness = 0
    while num_pixels is not None and current_num_pixels < num_pixels:
        thickness += 1
        buffer = binary_dilation(dilated_signal, iterations=thickness) & (~dilated_signal)
        current_num_pixels = np.sum(buffer)

    return buffer

