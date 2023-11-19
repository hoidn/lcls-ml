from pvalues import compute_aggregate_pvals
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
#from lcls import pvalues
import pvalues
compute_aggregate_pvals_with_custom_background = pvalues.compute_aggregate_pvals_with_custom_background

import histogram_analysis as hist
#from histogram_analysis import calculate_signal_background_noI0
calculate_signal_background_noI0 = hist.calculate_signal_background_noI0

def delay_bin(delay, delay_raw, Time_bin, arg_delay_nan):
    # Adjust the bin width to ensure it's a float
    Time_bin = float(Time_bin)

    # Determine the minimum and maximum values from the non-NaN delays
    delay_min = np.floor(delay_raw[arg_delay_nan == False].min())
    delay_max = np.ceil(delay_raw[arg_delay_nan == False].max())

    # Create bins that are shifted by half the bin width
    half_bin = Time_bin / 2
    bins = np.arange(delay_min - half_bin, delay_max + Time_bin, Time_bin)

    # Assign each delay to the nearest bin
    binned_indices = np.digitize(delay, bins, right=True)

    # Convert bin indices to delay values
    binned_delays = bins[binned_indices - 1] + half_bin

    # Ensure that the binned delays are within the min and max range
    binned_delays = np.clip(binned_delays, delay_min, delay_max)

    return binned_delays


def extract_stacks_by_delay_optimized(binned_delays, img_array, bin_width, min_count):
    unique_binned_delays = np.unique(binned_delays)
    stacks = {}

    mask = np.zeros_like(binned_delays, dtype=bool)
    for d in unique_binned_delays:
        mask |= (binned_delays == d)

    filtered_binned_delays = binned_delays[mask]
    filtered_imgs = img_array[mask]

    for d in unique_binned_delays:
        specific_mask = (filtered_binned_delays == d)
        stack = filtered_imgs[specific_mask]

        if stack.shape[0] >= min_count:
            stacks[d] = stack

    return stacks

def CDW_PP(Run_Number, ROI, Energy_Filter, I0_Threshold, IPM_pos_Filter, Time_bin, TimeTool,
          min_count = 200):
    from ybco import SMD_Loader, EnergyFilter
    rr = SMD_Loader(Run_Number)  # Small Data Import

    # Mask for bad pixels
    #idx_tile = rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0,0]
    #mask = rr.UserDataCfg.jungfrau1M.mask[idx_tile][rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,1],rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,1]]

    I0 = rr.ipm2.sum[:]
    arg_I0 = (I0 >= I0_Threshold)

    # IPM Positional Filter
    I0_x = rr.ipm2.xpos[:]
    I0_y = rr.ipm2.ypos[:]
    arg = (abs(I0_x) < 2.) & (abs(I0_y) < 3.)
    I0_x_mean, I0_y_mean = I0_x[arg].mean(), I0_y[arg].mean()
    arg_I0_x = (I0_x < (I0_x_mean + IPM_pos_Filter[0])) & (I0_x > (I0_x_mean - IPM_pos_Filter[0]))
    arg_I0_y = (I0_y < (I0_y_mean + IPM_pos_Filter[1])) & (I0_y > (I0_y_mean - IPM_pos_Filter[1]))

    # Time Tool Logic
    tt_arg = TimeTool[0]
    delay = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS) * tt_arg
    arg_delay_nan = np.isnan(delay)

    # Energy Filtering
    imgs = EnergyFilter(rr, Energy_Filter, ROI)

    # Laser On/Off Logic
    arg_laser_on = (np.array(rr.evr.code_90) == 1.)
    arg_laser_off = (np.array(rr.evr.code_91) == 1.)

    binned_delays = delay_bin(delay, np.array(rr.enc.lasDelay), Time_bin, arg_delay_nan)

    stacks_on = extract_stacks_by_delay_optimized(binned_delays[arg_laser_on], imgs[arg_laser_on], Time_bin, min_count)
    stacks_off = extract_stacks_by_delay_optimized(binned_delays[arg_laser_off], imgs[arg_laser_off], Time_bin, min_count)

    return {
    'stacks_on': stacks_on,
    'stacks_off': stacks_off,
    'I0': I0,
    'binned_delays': binned_delays,
    'arg_laser_on': arg_laser_on,
    'arg_laser_off': arg_laser_off
    }
    #return stacks_on, stacks_off, I0, binned_delays, arg_laser_on, arg_laser_off

def process_stacks(stacks, I0, arg_laser_condition, signal_mask, bin_boundaries, hist_start_bin,
                  binned_delays, background_mask_multiple= 1):
    delays, norm_signals, std_devs = [], [], []

    for delay, stack in stacks.items():
        # Filter I0 values for the specific delay and laser condition
        I0_filtered = I0[arg_laser_condition & (binned_delays == delay)]

        signal, bg, total_var = calculate_signal_background_noI0(stack, signal_mask, bin_boundaries, hist_start_bin,
                                                                background_mask_multiple= background_mask_multiple)
        norm_signal = (signal - bg) / np.mean(I0_filtered) if np.mean(I0_filtered) != 0 else 0
        std_dev = np.sqrt(total_var) / np.mean(I0_filtered) if np.mean(I0_filtered) != 0 else 0

        delays.append(delay)
        norm_signals.append(norm_signal)
        std_devs.append(std_dev)

    return delays, norm_signals, std_devs

def calculate_p_value_simplified(signal_on, signal_off, std_dev_on, std_dev_off):
    """
    Corrected p-value calculation using standard normal distribution.

    Args:
    signal_on (float): Signal for 'Laser On' condition.
    signal_off (float): Signal for 'Laser Off' condition.
    std_dev_on (float): Standard deviation for 'Laser On' condition.
    std_dev_off (float): Standard deviation for 'Laser Off' condition.

    Returns:
    float: Calculated p-value.
    """
#     # Calculating relative p-values
#     relative_p_values = []
#     for time_delay in sorted(stacks_on.keys()):
#         if time_delay in stacks_off:
#             size = min(stacks_on[time_delay].shape[0], stacks_off[time_delay].shape[0])
#             histo_on = calculate_histograms(stacks_on[time_delay][:size, ...], bin_boundaries, hist_start_bin)
#             histo_off = calculate_histograms(stacks_off[time_delay][:size, ...], bin_boundaries, hist_start_bin)
#             relative_histogram = np.abs(histo_on - histo_off)
#             p_value_data = compute_aggregate_pvals_with_custom_background(
#                 bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple,
#                 signal_mask=signal_mask, histograms=relative_histogram, num_permutations=10000
#             )
#             relative_p_values.append(p_value_data['aggregate_p_value'])
    from scipy.stats import norm
    delta_signal = abs(signal_on - signal_off)
    combined_std_dev = np.sqrt(std_dev_on**2 + std_dev_off**2)
    z_score = delta_signal / combined_std_dev

    # Using the CDF of the standard normal distribution to calculate p-value
    p_value = 2 * (1 - norm.cdf(z_score))  # Two-tailed test
    return p_value
