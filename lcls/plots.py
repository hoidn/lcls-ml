import matplotlib.pyplot as plt
import numpy as np

import pvalues
import histogram_analysis

compute_aggregate_pvals = pvalues.compute_aggregate_pvals
calculate_signal_background_noI0 = histogram_analysis.calculate_signal_background_noI0
calculate_histograms = histogram_analysis.calculate_histograms

compute_aggregate_pvals_with_custom_background = pvalues.compute_aggregate_pvals_with_custom_background
#from lcls.pvalues import compute_aggregate_pvals


from typing import Dict, List, Tuple, Any

#def plot_normalized_signal_vs_time_delay(
#    stacks_on: Dict[float, np.ndarray],
#    stacks_off: Dict[float, np.ndarray],
#    I0: np.ndarray,
#    arg_laser_on: np.ndarray,
#    arg_laser_off: np.ndarray,
#    signal_mask: np.ndarray,
#    bin_boundaries: np.ndarray,
#    hist_start_bin: int,
#    roi_coordinates: Tuple[int, int, int, int],
#    background_mask_multiple: float
#) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
#    """
#    Plots normalized signal vs time delay and aggregate p-values for 'Laser On' and 'Laser Off' scenarios, using provided image stacks.
#
#    Args:
#    stacks_on (Dict[float, np.ndarray]): Dictionary of time delays to image stacks for 'Laser On'.
#    stacks_off (Dict[float, np.ndarray]): Dictionary of time delays to image stacks for 'Laser Off'.
#    I0 (np.ndarray): Array of intensity values for normalization.
#    arg_laser_on (np.ndarray): Indices for 'Laser On' condition.
#    arg_laser_off (np.ndarray): Indices for 'Laser Off' condition.
#    signal_mask (np.ndarray): Boolean array representing the signal mask.
#    bin_boundaries (np.ndarray): Boundaries of the bins in the histogram.
#    hist_start_bin (int): Starting bin for histogram analysis.
#    roi_coordinates (Tuple[int, int, int, int]): Coordinates of the Region of Interest (ROI).
#    background_mask_multiple (float): Multiple of the number of pixels in the signal mask to approximate in the background mask.
#
#    Returns:
#    Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
#        A tuple containing lists of time delays, normalized signals, standard deviations, and aggregate p-values for 'Laser On' and 'Laser Off'.
#
#    The function processes each stack of images for 'Laser On' and 'Laser Off',
#    calculates the normalized signal and its standard deviation, and computes
#    aggregate p-values. It plots two subplots: one for the normalized signal vs
#    time delay and another for aggregate p-values vs time delay.
#    """
#    def process_stacks(stacks, I0_arg, signal_mask):
#        delays, norm_signals, std_devs, pvals = [], [], [], []
#
#        for delay, stack in stacks.items():
#            signal, bg, total_var = calculate_signal_background_noI0(stack, signal_mask, bin_boundaries, hist_start_bin)
#            norm_signal = (signal - bg) / np.mean(I0[I0_arg])
#            std_dev = np.sqrt(total_var) / np.mean(I0[I0_arg])
#
#            pval_result = compute_aggregate_pvals_with_custom_background(bin_boundaries, hist_start_bin,
#                                                                        roi_coordinates, background_mask_multiple,
#                                                                        signal_mask=signal_mask, data=stack)
#            pval = pval_result["aggregate_p_value"]
#
#            delays.append(delay)
#            norm_signals.append(norm_signal)
#            std_devs.append(std_dev)
#            pvals.append(pval)
#
#        return delays, norm_signals, std_devs, pvals

#def plot_normalized_signal_vs_time_delay(
#    stacks_on: Dict[float, np.ndarray],
#    stacks_off: Dict[float, np.ndarray],
#    I0: np.ndarray,
#    arg_laser_on: np.ndarray,
#    arg_laser_off: np.ndarray,
#    signal_mask: np.ndarray,
#    bin_boundaries: np.ndarray,
#    hist_start_bin: int,
#    roi_coordinates: Tuple[int, int, int, int],
#    background_mask_multiple: float
#):

def plot_normalized_signal_vs_time_delay(
    stacks_on: Dict[float, np.ndarray],
    stacks_off: Dict[float, np.ndarray],
    I0: np.ndarray,
    arg_laser_on: np.ndarray,
    arg_laser_off: np.ndarray,
    signal_mask: np.ndarray,
    bin_boundaries: np.ndarray,
    hist_start_bin: int,
    roi_coordinates: Tuple[int, int, int, int],
    background_mask_multiple: float,
    num_permutations: float = 1000
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Modified function to plot normalized signal vs time delay and relative aggregate p-values using provided image stacks.

    Args:
    stacks_on, stacks_off, I0, arg_laser_on, arg_laser_off, signal_mask, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple:
        Parameters as described in the original function documentation.

    Returns:
    Tuple containing lists of time delays, normalized signals, standard deviations, and relative aggregate p-values.
    """
    def process_stacks(stacks, I0_arg, signal_mask):
        delays, norm_signals, std_devs = [], [], []

        for delay, stack in stacks.items():
            signal, bg, total_var = calculate_signal_background_noI0(stack,
                            signal_mask, bin_boundaries, hist_start_bin,
                            background_mask_multiple= background_mask_multiple)
            norm_signal = (signal - bg) / np.mean(I0[I0_arg])
            std_dev = np.sqrt(total_var) / np.mean(I0[I0_arg])

            delays.append(delay)
            norm_signals.append(norm_signal)
            std_devs.append(std_dev)

        return delays, norm_signals, std_devs

    # Process 'Laser On' and 'Laser Off' stacks
    delays_on, norm_signal_on, std_dev_on = process_stacks(stacks_on, arg_laser_on, signal_mask)
    delays_off, norm_signal_off, std_dev_off = process_stacks(stacks_off, arg_laser_off, signal_mask)

    # Calculating relative p-values
    relative_p_values = []
    for time_delay in sorted(stacks_on.keys()):
        # TODO proper subsampling
        size = min(stacks_on[time_delay].shape[0], stacks_off[time_delay].shape[0])
        histo_on = calculate_histograms(stacks_on[time_delay][:size, ...], bin_boundaries, hist_start_bin)
        histo_off = calculate_histograms(stacks_off[time_delay][:size, ...], bin_boundaries, hist_start_bin)
        relative_histogram = np.abs(histo_on - histo_off)
        p_value_data = compute_aggregate_pvals_with_custom_background(
            bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple,
            signal_mask=signal_mask, histograms=relative_histogram, num_permutations=10000
        )
        relative_p_values.append(p_value_data['aggregate_p_value'])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot for Normalized Signal
    ax1.errorbar(delays_on, norm_signal_on, yerr=std_dev_on, fmt='rs-', label='Laser On: Signal')
    ax1.errorbar(delays_off, norm_signal_off, yerr=std_dev_off, fmt='ks-', mec='k', mfc='white', alpha=0.2, label='Laser Off: Signal')
    ax1.set_xlabel('Time Delay (ps)')
    ax1.set_ylabel('Normalized Signal')
    ax1.set_title('Normalized Signal vs Time Delay')
    ax1.legend()
    ax1.grid(True)
    ax1.minorticks_on()

    # Plot for Relative Aggregate P-Values
    ax2.plot(sorted(stacks_on.keys()), relative_p_values, 'm*--', label='Relative Aggregate P-Value')
    ax2.set_xlabel('Time Delay (ps)')
    ax2.set_ylabel('Relative Aggregate P-Value')
    ax2.set_title('Relative Aggregate P-Values vs Time Delay')
    ax2.legend()
    ax2.grid(True)
    ax2.minorticks_on()

    plt.show()

    return delays_on, norm_signal_on, std_dev_on, delays_off, norm_signal_off, std_dev_off, relative_p_values
