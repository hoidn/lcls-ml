#from pvalues import compute_aggregate_pvals
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
#from lcls import pvalues
import pvalues
compute_aggregate_pvals_with_custom_background = pvalues.compute_aggregate_pvals_with_custom_background

import histogram_analysis as hist
#from histogram_analysis import calculate_signal_background_noI0
calculate_signal_background_noI0 = hist.calculate_signal_background_noI0

from smd import SMD_Loader, EnergyFilter

# TODO: Implement a command line parameter to control the EnergyFilter behavior between filtering all energies above the third harmonic by default and selecting the old behavior.


from plots import geometric_mean
from stacks import generate_plot_data, process_stacks

def create_data_array(stacks_on, stacks_off):
    """
    Creates a 3D numpy array by stacking the 2D numpy arrays in stacks_on and stacks_off along the 0th axis.

    :param stacks_on: A dictionary of 2D numpy arrays representing 'on' data.
    :param stacks_off: A dictionary of 2D numpy arrays representing 'off' data.
    :return: A 3D numpy array created by stacking the arrays in stacks_on and stacks_off along the 0th axis.
    """
    on_arrays = [stack for key in sorted(stacks_on.keys()) for stack in stacks_on[key]]
    off_arrays = [stack for key in sorted(stacks_off.keys()) for stack in stacks_off[key]]

    # Stacking all arrays along the 0th axis into a single 3D numpy array
    data = np.stack(on_arrays + off_arrays, axis=0)
    return data

from histogram_analysis import run_histogram_analysis
# TODO signal mask from histograms
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
            # Calculate 'data'
            data = create_data_array(cdw_output['stacks_on'], cdw_output['stacks_off'])
        else:
            # Calculate 'histograms'
            histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)

    # Run histogram analysis to get the signal mask
    analysis_results = run_histogram_analysis(histograms=histograms, bin_boundaries=bin_boundaries,
                                              hist_start_bin=hist_start_bin, roi_x_start=roi_x_start,
                                              roi_x_end=roi_x_end, roi_y_start=roi_y_start, roi_y_end=roi_y_end,
                                              threshold=threshold)
    signal_mask = analysis_results['signal_mask']

    # Run the analysis/visualization
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

    p_values = analysis_results['relative_p_values']  # Extract p-values from the results
    p_values = np.array(p_values)
    p_values = p_values[p_values > 0]  # Exclude non-positive values for geometric mean calculation

    if len(p_values) == 0:
        return 0  # Return 0 if there are no positive p-values

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
            # Run analysis with current set of parameters
            analysis_results = run_analysis_and_visualization(
                cdw_output, bin_boundaries, hist_start_bin, roi_coordinates,
                background_mask_multiple, threshold, histograms= histograms
            )

            # Calculate the figure of merit for the current parameter set
            current_figure_of_merit = calculate_figure_of_merit(analysis_results)

            # Update the optimal parameters if the current figure of merit is better
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

from deps import create_background_mask
def optimize_signal_mask(bin_boundaries, hist_start_bin, roi_coordinates, histograms,
                         threshold_lower=0., threshold_upper=.5,
                         num_threshold_points=10, num_runs=5,
                         max_signal_fraction = 0.4):
    """
    Performs grid search optimization for the 'signal mask' array, focusing only on the threshold parameter.
    """
    threshold_range = np.linspace(threshold_lower, threshold_upper, num_threshold_points)
    best_signal_mask = None
    best_threshold = None
    best_avg_ratio = float('-inf')
    grid_search_results = []

    for threshold in threshold_range:
        ratios = []

        for _ in range(num_runs):
            signal_mask = compute_signal_mask(bin_boundaries, hist_start_bin, roi_coordinates, threshold, histograms=histograms)
            print(f"Threshold: {threshold}, Signal mask mean: {signal_mask.mean()}, Signal mask sum: {signal_mask.sum()}")
            if signal_mask.sum() == 0:
                print("Skipping due to signal mask sum == 0")
            elif signal_mask.mean() > max_signal_fraction:
                print(f"Skipping due to signal mask mean > max_signal_fraction ({max_signal_fraction})")
                continue

            background_mask = create_background_mask(signal_mask, 1, 10)
            signal, bg, _ = hist.calculate_signal_background_from_histograms(histograms, signal_mask, background_mask, bin_boundaries, hist_start_bin)

            if bg == 0:
                print("Skipping due to background == 0")
                continue

            ratio = (signal - bg) / bg
            ratios.append(ratio)

        if not ratios:
            print("No ratios calculated, continuing to next threshold.")
            continue

        avg_ratio = np.mean(ratios)
        std_dev = np.std(ratios)
        grid_search_results.append((threshold, avg_ratio, std_dev))

        if avg_ratio > best_avg_ratio:
            best_avg_ratio = avg_ratio
            best_signal_mask = signal_mask
            best_threshold = threshold

    # Converting grid search results to a numpy array or dict of numpy arrays
    grid_search_results_np = np.array(grid_search_results, dtype=[('threshold', float), ('avg_ratio', float), ('std_dev', float)])

    return best_signal_mask, best_threshold, grid_search_results_np

# Note: The functions compute_signal_mask and calculate_signal_background_from_histograms need to be predefined as per the user's environment.



import matplotlib.gridspec as gridspec

def plot_data(data, subplot_spec=None, plot_title='Normalized Signal vs Time Delay', save_path='plot.png'):
    fig = plt.gcf()

    if subplot_spec is None:
        gs = gridspec.GridSpec(2, 1)  # Default to 2x1 grid
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
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
    ax2.set_title('FOM: {:.2f}'.format(-np.log10(geometric_mean(relative_p_values))))
    #ax2.set_title('FOM: {:.2f}'.format(1 - geometric_mean(relative_p_values)))

    #ax2.set_ylim(0, 4)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(save_path)  # Save the figure to a file
    plt.show()  # Display the figure

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
        p_value = 2 * (1 - norm.cdf(z_score))  # Two-tailed test
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

    # Calculate relative p-values
    p_values = calculate_relative_p_values(Intensity_on, Intensity_off, assume_photon_counts)

    return {
        'delay': delay,
        'Intensity_on': Intensity_on,
        'Intensity_off': Intensity_off,
        'p_values': p_values
    }

from matplotlib.gridspec import GridSpec
def combine_plots(pp_lazy_data, cdw_data):
    # Create a figure with a 2x2 grid of subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)

    # First pair of plots
    plot_data_dict1 = {
        'delays_on': pp_lazy_data['delay'],
        'norm_signal_on': pp_lazy_data['Intensity_on'][:, 0],
        'std_dev_on': pp_lazy_data['Intensity_on'][:, 1],
        'delays_off': pp_lazy_data['delay'],
        'norm_signal_off': pp_lazy_data['Intensity_off'][:, 0],
        'std_dev_off': pp_lazy_data['Intensity_off'][:, 1],
        'relative_p_values': pp_lazy_data['p_values']
    }
    plot_data(plot_data_dict1, subplot_spec=[gs[0, 1], gs[1, 1]], plot_title = 'Human')

    # Second pair of plots
    plot_data(cdw_data, subplot_spec=[gs[0, 0], gs[1, 0]], plot_title = 'Automated')

    plt.tight_layout()
    plt.show()


def combine_plots_nopp(cdw_data, human_data):
    # Create a figure with a 2x2 grid of subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)


    plot_data(cdw_data, subplot_spec=[gs[0, 0], gs[1, 0]], plot_title = 'Automated')
    plot_data(human_data, subplot_spec=[gs[0, 1], gs[1, 1]], plot_title = 'Human')

    plt.tight_layout()
    plt.show()
