# TODO rename this module
from typing import Union
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
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

#def process_stacks(stacks, I0, arg_laser_condition, signal_mask, bin_boundaries, hist_start_bin, binned_delays, background_mask, subtract_background=True):
#    delays, norm_signals, std_devs = [], [], []
#    for delay, stack in stacks.items():
#        I0_filtered = I0[arg_laser_condition & (binned_delays == delay)]
#        signal, bg, total_var = calculate_signal_background_noI0(stack, signal_mask, bin_boundaries, hist_start_bin, background_mask)
#        if subtract_background:
#            norm_signal = (signal - bg) / np.mean(I0_filtered) if np.mean(I0_filtered) != 0 else 0
#        else:
#            norm_signal = signal / np.mean(I0_filtered) if np.mean(I0_filtered) != 0 else 0
#        std_dev = np.sqrt(total_var) / np.mean(I0_filtered) if np.mean(I0_filtered) != 0 else 0
#        delays.append(delay)
#        norm_signals.append(norm_signal)
#        std_devs.append(std_dev)
#    return delays, norm_signals, std_devs

def calculate_p_value(signal_on, signal_off, std_dev_on, std_dev_off):
    delta_signal = abs(signal_on - signal_off)
    combined_std_dev = np.sqrt(std_dev_on**2 + std_dev_off**2)
    z_score = delta_signal / combined_std_dev
    p_value = 2 * (1 - norm.cdf(z_score))
    return p_value

def generate_plot_data(cdw_pp_output, signal_mask, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple, subtract_background=True, thickness=10, separator_thickness=5):
    stacks_on = cdw_pp_output['stacks_on']
    stacks_off = cdw_pp_output['stacks_off']
    I0 = cdw_pp_output['I0']
    binned_delays = cdw_pp_output['binned_delays']
    arg_laser_on = cdw_pp_output['arg_laser_on']
    arg_laser_off = cdw_pp_output['arg_laser_off']
    background_mask = create_background_mask(signal_mask, background_mask_multiple, thickness)
    delays_on, norm_signal_on, std_dev_on = process_stacks(stacks_on, I0, arg_laser_on, signal_mask, bin_boundaries, hist_start_bin, binned_delays, background_mask, subtract_background=subtract_background)
    delays_off, norm_signal_off, std_dev_off = process_stacks(stacks_off, I0, arg_laser_off, signal_mask, bin_boundaries, hist_start_bin, binned_delays, background_mask, subtract_background=subtract_background)
    relative_p_values = []
    for delay in sorted(set(delays_on) & set(delays_off)):
        signal_on = norm_signal_on[delays_on.index(delay)]
        signal_off = norm_signal_off[delays_off.index(delay)]
        std_dev_on_val = std_dev_on[delays_on.index(delay)]
        std_dev_off_val = std_dev_off[delays_off.index(delay)]
        p_value = calculate_p_value(signal_on, signal_off, std_dev_on_val, std_dev_off_val)
        relative_p_values.append(p_value)
    return {
        'delays_on': delays_on,
        'norm_signal_on': norm_signal_on,
        'std_dev_on': std_dev_on,
        'delays_off': delays_off,
        'norm_signal_off': norm_signal_off,
        'std_dev_off': std_dev_off,
        'relative_p_values': relative_p_values,
        'background_mask': background_mask
    }

    return {
        'delays_on': delays_on,
        'norm_signal_on': norm_signal_on,
        'std_dev_on': std_dev_on,
        'delays_off': delays_off,
        'norm_signal_off': norm_signal_off,
        'std_dev_off': std_dev_off,
        'relative_p_values': relative_p_values
    }

def calculate_signal_background_noI0(data: np.ndarray, signal_mask: np.ndarray, bin_boundaries: np.ndarray, hist_start_bin: int, background_mask: np.ndarray, **kwargs):
    assert isinstance(data, np.ndarray), "data must be a numpy array"
    assert isinstance(signal_mask, np.ndarray), "signal_mask must be a numpy array"
    assert isinstance(bin_boundaries, np.ndarray), "bin_boundaries must be a numpy array"
    assert isinstance(hist_start_bin, int), "hist_start_bin must be an integer"
    assert isinstance(background_mask, np.ndarray), "background_mask must be a numpy array"
    local_histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)
    return calculate_signal_background_from_histograms(local_histograms, signal_mask, background_mask, bin_boundaries, hist_start_bin, **kwargs)

from numba import jit
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



def calculate_total_counts(integrated_counts: np.ndarray, signal_mask: np.ndarray) -> int:
    counts_in_signal = integrated_counts[signal_mask]
    S = np.sum(counts_in_signal)
    return S

def background_subtraction(integrated_counts: np.ndarray, signal_mask: np.ndarray, buffer: np.ndarray) -> Tuple[float, float]:
    # TODO unequal signal and background
    N = np.sum(signal_mask)
    M = calculate_total_counts(integrated_counts, buffer)
    if M is None:
        return None
    S = calculate_total_counts(integrated_counts, signal_mask)
    return S - (M * N / np.sum(buffer)), M * N / np.sum(buffer)

def calculate_signal_background_from_histograms(local_histograms, signal_mask, background_mask, bin_boundaries, hist_start_bin, Emin = 8, Emax = 10):
    energies = bin_boundaries[hist_start_bin + 1:]
    integrated_counts = filter_and_sum_histograms(local_histograms, energies, Emin, Emax)
    signal, bg = background_subtraction(integrated_counts, signal_mask, background_mask)
    var_signal = signal
    var_bg = bg
    nsignal = np.sum(signal_mask)
    nbg = np.sum(background_mask)
    total_var = var_signal + (var_bg * ((nsignal / nbg)**2))
    return signal, bg, total_var


def create_background_mask(signal_mask, background_mask_multiple, thickness, separator_thickness = 5):
    num_pixels_signal_mask = np.sum(signal_mask)
    num_pixels_background_mask = int(num_pixels_signal_mask * background_mask_multiple)
    buffer = create_continuous_buffer(signal_mask,
                initial_thickness=thickness, num_pixels=num_pixels_background_mask,
                                               separator_thickness=separator_thickness)
    return buffer


def filter_and_sum_histograms(histograms, energies, Emin, Emax):
    E0 = (Emax + Emin) / 2
    window_size = Emax - Emin
    energy_mask = np.zeros_like(energies, dtype=bool)
    for n in range(1, 4):
        harmonic_window_size = np.sqrt(n) * window_size
        harmonic_center_energy = n * E0
        harmonic_Emin = harmonic_center_energy - harmonic_window_size / 2
        harmonic_Emax = harmonic_center_energy + harmonic_window_size / 2
        energy_mask |= ((energies >= harmonic_Emin) & (energies <= harmonic_Emax))
    filtered_histograms = histograms[energy_mask, :, :]
    summed_histograms = np.sum(filtered_histograms, axis=0)
    return summed_histograms


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
        buffer = binary_dilation(dilated_signal, iterations=thickness) & (~signal_mask)
        current_num_pixels = np.sum(buffer)

    return buffer
