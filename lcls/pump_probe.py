from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pvalues
compute_aggregate_pvals_with_custom_background = pvalues.compute_aggregate_pvals_with_custom_background

import histogram_analysis as hist
calculate_signal_background_noI0 = hist.calculate_signal_background_noI0

def delay_bin(delay, delay_raw, Time_bin, arg_delay_nan):
    Time_bin = float(Time_bin)

    delay_min = np.floor(delay_raw[arg_delay_nan == False].min())
    delay_max = np.ceil(delay_raw[arg_delay_nan == False].max())

    half_bin = Time_bin / 2
    bins = np.arange(delay_min - half_bin, delay_max + Time_bin, Time_bin)

    binned_indices = np.digitize(delay, bins, right=True)

    binned_delays = bins[binned_indices - 1] + half_bin

    binned_delays = np.clip(binned_delays, delay_min, delay_max)

    return binned_delays


def extract_stacks_by_delay(binned_delays, img_array, bin_width, min_count):
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
    rr = SMD_Loader(Run_Number)  


    I0 = rr.ipm2.sum[:]
    arg_I0 = (I0 >= I0_Threshold)

    I0_x = rr.ipm2.xpos[:]
    I0_y = rr.ipm2.ypos[:]
    arg = (abs(I0_x) < 2.) & (abs(I0_y) < 3.)
    I0_x_mean, I0_y_mean = I0_x[arg].mean(), I0_y[arg].mean()
    arg_I0_x = (I0_x < (I0_x_mean + IPM_pos_Filter[0])) & (I0_x > (I0_x_mean - IPM_pos_Filter[0]))
    arg_I0_y = (I0_y < (I0_y_mean + IPM_pos_Filter[1])) & (I0_y > (I0_y_mean - IPM_pos_Filter[1]))

    tt_arg = TimeTool[0]
    delay = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS) * tt_arg
    arg_delay_nan = np.isnan(delay)

    imgs = EnergyFilter(rr, Energy_Filter, ROI)

    arg_laser_on = (np.array(rr.evr.code_90) == 1.)
    arg_laser_off = (np.array(rr.evr.code_91) == 1.)

    binned_delays = delay_bin(delay, np.array(rr.enc.lasDelay), Time_bin, arg_delay_nan)

    stacks_on = extract_stacks_by_delay(binned_delays[arg_laser_on], imgs[arg_laser_on], Time_bin, min_count)
    stacks_off = extract_stacks_by_delay(binned_delays[arg_laser_off], imgs[arg_laser_off], Time_bin, min_count)

    return {
    'stacks_on': stacks_on,
    'stacks_off': stacks_off,
    'I0': I0,
    'binned_delays': binned_delays,
    'arg_laser_on': arg_laser_on,
    'arg_laser_off': arg_laser_off
    }

def process_stacks(stacks, I0, arg_laser_condition, signal_mask, bin_boundaries, hist_start_bin,
                  binned_delays, background_mask_multiple= 1):
    delays, norm_signals, std_devs = [], [], []

    for delay, stack in stacks.items():
        I0_filtered = I0[arg_laser_condition & (binned_delays == delay)]

        signal, bg, total_var = calculate_signal_background_noI0(stack, signal_mask, bin_boundaries, hist_start_bin,
                                                                background_mask_multiple= background_mask_multiple)
        norm_signal = (signal - bg) / np.mean(I0_filtered) if np.mean(I0_filtered) != 0 else 0
        std_dev = np.sqrt(total_var) / np.mean(I0_filtered) if np.mean(I0_filtered) != 0 else 0

        delays.append(delay)
        norm_signals.append(norm_signal)
        std_devs.append(std_dev)

    return delays, norm_signals, std_devs

def calculate_p_value(signal_on, signal_off, std_dev_on, std_dev_off):
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
    from scipy.stats import norm
    delta_signal = abs(signal_on - signal_off)
    combined_std_dev = np.sqrt(std_dev_on**2 + std_dev_off**2)
    z_score = delta_signal / combined_std_dev

    p_value = 2 * (1 - norm.cdf(z_score))  
    return p_value

from plots import geometric_mean

def create_data_array(stacks_on, stacks_off):
    """
    Creates a 3D numpy array by stacking the 2D numpy arrays in stacks_on and stacks_off along the 0th axis.

    :param stacks_on: A dictionary of 2D numpy arrays representing 'on' data.
    :param stacks_off: A dictionary of 2D numpy arrays representing 'off' data.
    :return: A 3D numpy array created by stacking the arrays in stacks_on and stacks_off along the 0th axis.
    """
    on_arrays = [stack for key in sorted(stacks_on.keys()) for stack in stacks_on[key]]
    off_arrays = [stack for key in sorted(stacks_off.keys()) for stack in stacks_off[key]]

    data = np.stack(on_arrays + off_arrays, axis=0)
    return data

from histogram_analysis import run_histogram_analysis
def compute_signal_mask(bin_boundaries, hist_start_bin, roi_coordinates, threshold,
                                data = None, histograms = None):
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

def run_analysis_and_visualization(cdw_output, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple, threshold,
                                  data = None, histograms = None):
    """
    Top-level function to run analysis and visualization.

    :param cdw_output: Dictionary containing 'stacks_on' and 'stacks_off'.
    :param bin_boundaries: Boundaries for histogram bins.
    :param hist_start_bin: The starting bin for the histogram.
    :param roi_coordinates: ROI coordinates as a tuple (roi_x_start, roi_x_end, roi_y_start, roi_y_end).
    :param background_mask_multiple: Parameter for the plot_normalized_signal_vs_time_delay function.
    :param threshold: Threshold value for signal mask calculation.
    :return: Dictionary capturing the outputs of the plot_normalized_signal_vs_time_delay function.
    """
    if histograms is None:
        if data is None:
            data = create_data_array(cdw_output['stacks_on'], cdw_output['stacks_off'])
        else:
            histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)

    analysis_results = run_histogram_analysis(histograms=histograms, bin_boundaries=bin_boundaries,
                                              hist_start_bin=hist_start_bin, roi_x_start=roi_x_start,
                                              roi_x_end=roi_x_end, roi_y_start=roi_y_start, roi_y_end=roi_y_end,
                                              threshold=threshold)
    signal_mask = analysis_results['signal_mask']

    plot_results = plot_normalized_signal_vs_time_delay(cdw_output, signal_mask,
                                                        bin_boundaries, hist_start_bin,
                                                        roi_coordinates, background_mask_multiple)

    return plot_results

def calculate_figure_of_merit(analysis_results):
    """
    Calculates the geometric mean of the p-values from the analysis results.

    :param analysis_results: Dictionary containing analysis results with p-values.
    :return: Geometric mean of the p-values.
    """
    import numpy as np

    p_values = analysis_results['relative_p_values']  
    p_values = np.array(p_values)
    p_values = p_values[p_values > 0]  

    if len(p_values) == 0:
        return 0  

    geometric_mean = np.exp(np.mean(np.log(p_values)))
    return 1 - geometric_mean

def optimize_figure_of_merit(cdw_output, bin_boundaries, hist_start_bin, roi_coordinates, param_grid,
                            histograms = None):
    """
    Optimizes the figure of merit over analysis parameters using grid search.

    :param cdw_output: Dictionary containing 'stacks_on' and 'stacks_off'.
    :param bin_boundaries: Boundaries for histogram bins.
    :param hist_start_bin: The starting bin for the histogram.
    :param roi_coordinates: ROI coordinates as a tuple (roi_x_start, roi_x_end, roi_y_start, roi_y_end).
    :param param_grid: Dictionary defining the grid search space for 'background_mask_multiple' and 'threshold'.
    :return: Dictionary containing the optimal parameters and the corresponding figure of merit.
    """
    optimal_params = None
    best_figure_of_merit = float('-inf')

    for background_mask_multiple in param_grid['background_mask_multiple']:
        for threshold in param_grid['threshold']:
            print(background_mask_multiple, threshold)
            analysis_results = run_analysis_and_visualization(
                cdw_output, bin_boundaries, hist_start_bin, roi_coordinates,
                background_mask_multiple, threshold, histograms= histograms
            )

            current_figure_of_merit = calculate_figure_of_merit(analysis_results)

            if current_figure_of_merit > best_figure_of_merit:
                best_figure_of_merit = current_figure_of_merit
                optimal_params = {
                    'background_mask_multiple': background_mask_multiple,
                    'threshold': threshold
                }

    return {
        'optimal_params': optimal_params,
        'best_figure_of_merit': best_figure_of_merit
    }


def generate_plot_data(cdw_pp_output, signal_mask, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple):
    stacks_on = cdw_pp_output['stacks_on']
    stacks_off = cdw_pp_output['stacks_off']
    I0 = cdw_pp_output['I0']
    binned_delays = cdw_pp_output['binned_delays']
    arg_laser_on = cdw_pp_output['arg_laser_on']
    arg_laser_off = cdw_pp_output['arg_laser_off']

    delays_on, norm_signal_on, std_dev_on = process_stacks(stacks_on, I0, arg_laser_on, signal_mask,
            bin_boundaries, hist_start_bin, binned_delays, background_mask_multiple=background_mask_multiple)
    delays_off, norm_signal_off, std_dev_off = process_stacks(stacks_off, I0, arg_laser_off, signal_mask,
            bin_boundaries, hist_start_bin, binned_delays, background_mask_multiple=background_mask_multiple)

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
        'relative_p_values': relative_p_values
    }


def plot_data(data, subplot_spec, plot_title = 'Normalized Signal vs Time Delay'):
    fig = plt.gcf()  
    ax1 = fig.add_subplot(subplot_spec[0])
    ax2 = fig.add_subplot(subplot_spec[1])

    delays_on = data['delays_on']
    norm_signal_on = data['norm_signal_on']
    std_dev_on = data['std_dev_on']
    delays_off = data['delays_off']
    norm_signal_off = data['norm_signal_off']
    std_dev_off = data['std_dev_off']
    relative_p_values = data['relative_p_values']

    ax1.errorbar(delays_on, norm_signal_on, yerr=std_dev_on, fmt='rs-', label='Laser On: Signal')
    ax1.errorbar(delays_off, norm_signal_off, yerr=std_dev_off, fmt='ks-', mec='k', mfc='white', alpha=0.2, label='Laser Off: Signal')
    ax1.set_xlabel('Time Delay (ps)')
    ax1.set_ylabel('Normalized Signal')
    ax1.set_title(plot_title)
    ax1.legend()
    ax1.grid(True)
    ax1.minorticks_on()

    neg_log_p_values = [-np.log10(p) if p > 0 else 0 for p in relative_p_values]
    ax2.set_xlabel('Time Delay')
    ax2.set_ylabel('-log(P-value)')
    ax2.set_title('-log(P-value) vs Time Delay')
    ax2.grid(True)
    ax2.scatter(sorted(set(delays_on) & set(delays_off)), neg_log_p_values, color='red', label='-log(p-value)')

    label_offset = 0.2  
    for p_val, label in zip([0.5, 0.1, 0.01, 0.001], ['50%', '10%', '1%', '0.1%']):
        neg_log_p_val = -np.log10(p_val)
        ax2.axhline(y=neg_log_p_val, color='black', linestyle='--')
        ax2.text(ax2.get_xlim()[1], neg_log_p_val + label_offset, f'{label} level', va='center', ha='right', fontsize=18, color='black')

    ax2.legend()
    ax2.set_title('FOM: {:.2f}'.format(1 - geometric_mean(relative_p_values)))
    ax2.set_ylim(0, 4)

def plot_normalized_signal_vs_time_delay(cdw_pp_output, signal_mask, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple):
    plot_data_dict = generate_plot_data(cdw_pp_output, signal_mask, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple)
    plot_data(plot_data_dict)
    return plot_data_dict

from scipy.stats import norm
def calculate_relative_p_values(Intensity_on, Intensity_off, assume_photon_counts=True):
    p_values = []
    for i in range(len(Intensity_on)):
        signal_on = Intensity_on[i, 0]
        signal_off = Intensity_off[i, 0]

        if assume_photon_counts:
            std_dev_on = np.sqrt(signal_on)
            std_dev_off = np.sqrt(signal_off)
        else:
            std_dev_on = Intensity_on[i, 1]
            std_dev_off = Intensity_off[i, 1]

        delta_signal = abs(signal_on - signal_off)
        combined_std_dev = np.sqrt(std_dev_on**2 + std_dev_off**2)
        z_score = delta_signal / combined_std_dev
        p_value = 2 * (1 - norm.cdf(z_score))  
        p_values.append(p_value)
    return np.array(p_values)

def generate_pp_lazy_data(imgs_on, imgs_off, mask, delay, assume_photon_counts=False):
    Intensity_on, Intensity_off = [], []
    npixels = (mask == 1).sum()
    for i in range(imgs_on.shape[0]):
        Intensity_on.append(imgs_on[i][mask == 1].mean())
        Intensity_on.append(imgs_on[i][mask == 1].std() / np.sqrt(npixels))
        Intensity_off.append(imgs_off[i][mask == 1].mean())
        Intensity_off.append(imgs_off[i][mask == 1].std() / np.sqrt(npixels))

    Intensity_on = np.array(Intensity_on).reshape(imgs_on.shape[0], 2)
    Intensity_off = np.array(Intensity_off).reshape(imgs_on.shape[0], 2)

    p_values = calculate_relative_p_values(Intensity_on, Intensity_off, assume_photon_counts)

    return {
        'delay': delay,
        'Intensity_on': Intensity_on,
        'Intensity_off': Intensity_off,
        'p_values': p_values
    }


from matplotlib.gridspec import GridSpec
def combine_plots(pp_lazy_data, cdw_data):
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)

    plot_data_dict1 = {
        'delays_on': pp_lazy_data['delay'],
        'norm_signal_on': pp_lazy_data['Intensity_on'][:, 0],
        'std_dev_on': pp_lazy_data['Intensity_on'][:, 1],
        'delays_off': pp_lazy_data['delay'],
        'norm_signal_off': pp_lazy_data['Intensity_off'][:, 0],
        'std_dev_off': pp_lazy_data['Intensity_off'][:, 1],
        'relative_p_values': pp_lazy_data['p_values']
    }
    plot_data(plot_data_dict1, subplot_spec=[gs[0, 0], gs[1, 0]], plot_title = 'Human')

    plot_data(cdw_data, subplot_spec=[gs[0, 1], gs[1, 1]], plot_title = 'Automated')

    plt.tight_layout()
    plt.show()
