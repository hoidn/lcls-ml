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
#):

def geometric_mean(arr):
    """
    Calculate the geometric mean of a numpy array.

    :param arr: numpy array
    :return: geometric mean of the array
    """
    return np.exp(np.mean(np.log(arr)))

from pump_probe import process_stacks, calculate_p_value_simplified
# Updating the plotting function to use this simplified p-value calculation
def plot_normalized_signal_vs_time_delay(
    cdw_pp_output: dict,
    signal_mask: np.ndarray,
    bin_boundaries: np.ndarray,
    hist_start_bin: int,
    roi_coordinates: Tuple[int, int, int, int],
    background_mask_multiple: float
):
    # Extracting data from CDW_PP output
    stacks_on = cdw_pp_output['stacks_on']
    stacks_off = cdw_pp_output['stacks_off']
    I0 = cdw_pp_output['I0']
    binned_delays = cdw_pp_output['binned_delays']
    arg_laser_on = cdw_pp_output['arg_laser_on']
    arg_laser_off = cdw_pp_output['arg_laser_off']

    # Process 'Laser On' and 'Laser Off' stacks
    delays_on, norm_signal_on, std_dev_on = process_stacks(stacks_on, I0, arg_laser_on, signal_mask,
            bin_boundaries, hist_start_bin, binned_delays, background_mask_multiple= background_mask_multiple)
    delays_off, norm_signal_off, std_dev_off = process_stacks(stacks_off, I0, arg_laser_off, signal_mask,
            bin_boundaries, hist_start_bin, binned_delays, background_mask_multiple= background_mask_multiple)

    # Calculating relative p-values
    relative_p_values = []
    for delay in sorted(set(delays_on) & set(delays_off)):
        signal_on = norm_signal_on[delays_on.index(delay)]
        signal_off = norm_signal_off[delays_off.index(delay)]
        std_dev_on_val = std_dev_on[delays_on.index(delay)]
        std_dev_off_val = std_dev_off[delays_off.index(delay)]

        p_value = calculate_p_value_simplified(signal_on, signal_off, std_dev_on_val, std_dev_off_val)
        relative_p_values.append(p_value)

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

    # Calculate -log(p-value) while handling cases where p-value is 0
    neg_log_p_values = [-np.log10(p) if p > 0 else 0 for p in relative_p_values]
    ax2.set_xlabel('Time Delay')
    ax2.set_ylabel('-log(P-value)')
    ax2.set_title('-log(P-value) vs Time Delay')
    ax2.legend()
    ax2.grid(True)
    # Plot for Relative Aggregate P-Values
    ax2.scatter(sorted(set(delays_on) & set(delays_off)), neg_log_p_values, color='red', label='-log(p-value)')
    # Adding dashed lines for 10%, 1%, and 0.1% levels
    label_offset = 0.2  # Adjust this value as needed for proper label positioning
    for p_val, label in zip([0.1, 0.01, 0.001], ['10%', '1%', '0.1%']):
        neg_log_p_val = -np.log10(p_val)
        ax2.axhline(y=neg_log_p_val, color='black', linestyle='--')
        ax2.text(ax2.get_xlim()[1], neg_log_p_val + label_offset, f'{label} level', va='center', ha='right', fontsize=18, color='black')

    #ax2.axhline(y=-np.log10(0.5), color='black', linestyle='--')
    ax2.set_xlabel('Time Delay (ps)')
    ax2.set_title('geometric mean: {}'.format(geometric_mean(relative_p_values)))
    ax2.minorticks_on()

    plt.show()

    return {
        'delays_on': delays_on,
        'norm_signal_on': norm_signal_on,
        'std_dev_on': std_dev_on,
        'delays_off': delays_off,
        'norm_signal_off': norm_signal_off,
        'std_dev_off': std_dev_off,
        'relative_p_values': relative_p_values
    }
    #return delays_on, norm_signal_on, std_dev_on, delays_off, norm_signal_off, std_dev_off, relative_p_values

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
#    background_mask_multiple: float,
#    num_permutations: float = 1000
#) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
#    """
#    Modified function to plot normalized signal vs time delay and relative aggregate p-values using provided image stacks.
#
#    Args:
#    stacks_on, stacks_off, I0, arg_laser_on, arg_laser_off, signal_mask, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple:
#        Parameters as described in the original function documentation.
#
#    Returns:
#    Tuple containing lists of time delays, normalized signals, standard deviations, and relative aggregate p-values.
#    """
#    def process_stacks(stacks, I0_arg, signal_mask):
#        delays, norm_signals, std_devs = [], [], []
#
#        for delay, stack in stacks.items():
#            signal, bg, total_var = calculate_signal_background_noI0(stack,
#                            signal_mask, bin_boundaries, hist_start_bin,
#                            background_mask_multiple= background_mask_multiple)
#            norm_signal = (signal - bg) / np.mean(I0[I0_arg])
#            std_dev = np.sqrt(total_var) / np.mean(I0[I0_arg])
#
#            delays.append(delay)
#            norm_signals.append(norm_signal)
#            std_devs.append(std_dev)
#
#        return delays, norm_signals, std_devs
#
#    # Process 'Laser On' and 'Laser Off' stacks
#    delays_on, norm_signal_on, std_dev_on = process_stacks(stacks_on, arg_laser_on, signal_mask)
#    delays_off, norm_signal_off, std_dev_off = process_stacks(stacks_off, arg_laser_off, signal_mask)
#
#    # Calculating relative p-values
#    relative_p_values = []
#    for time_delay in sorted(stacks_on.keys()):
#        # TODO proper subsampling
#        size = min(stacks_on[time_delay].shape[0], stacks_off[time_delay].shape[0])
#        histo_on = calculate_histograms(stacks_on[time_delay][:size, ...], bin_boundaries, hist_start_bin)
#        histo_off = calculate_histograms(stacks_off[time_delay][:size, ...], bin_boundaries, hist_start_bin)
#        relative_histogram = np.abs(histo_on - histo_off)
#        p_value_data = compute_aggregate_pvals_with_custom_background(
#            bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple,
#            signal_mask=signal_mask, histograms=relative_histogram, num_permutations=10000
#        )
#        relative_p_values.append(p_value_data['aggregate_p_value'])
#
#    # Plotting
#    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
#
#    # Plot for Normalized Signal
#    ax1.errorbar(delays_on, norm_signal_on, yerr=std_dev_on, fmt='rs-', label='Laser On: Signal')
#    ax1.errorbar(delays_off, norm_signal_off, yerr=std_dev_off, fmt='ks-', mec='k', mfc='white', alpha=0.2, label='Laser Off: Signal')
#    ax1.set_xlabel('Time Delay (ps)')
#    ax1.set_ylabel('Normalized Signal')
#    ax1.set_title('Normalized Signal vs Time Delay')
#    ax1.legend()
#    ax1.grid(True)
#    ax1.minorticks_on()
#
#    # Plot for Relative Aggregate P-Values
#    ax2.plot(sorted(stacks_on.keys()), relative_p_values, 'm*--', label='Relative Aggregate P-Value')
#    ax2.set_xlabel('Time Delay (ps)')
#    ax2.set_ylabel('Relative Aggregate P-Value')
#    ax2.set_title('Relative Aggregate P-Values vs Time Delay')
#    ax2.legend()
#    ax2.grid(True)
#    ax2.minorticks_on()
#
#    plt.show()
#
#    return delays_on, norm_signal_on, std_dev_on, delays_off, norm_signal_off, std_dev_off, relative_p_values
