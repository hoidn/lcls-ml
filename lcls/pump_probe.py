from lcls.visualization import plot_data, plot_normalized_signal_vs_time_delay, combine_plots, combine_plots_nopp
from lcls.statistics import calculate_relative_p_values, generate_pp_lazy_data
from lcls.masks import compute_signal_mask, create_background_mask

from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pvalues
compute_aggregate_pvals_with_custom_background = pvalues.compute_aggregate_pvals_with_custom_background

from lcls.deps import calculate_signal_background_noI0

# TODO: Implement a command line parameter to control the EnergyFilter behavior between filtering all energies above the third harmonic by default and selecting the old behavior.

from lcls.visualization import plot_data, plot_normalized_signal_vs_time_delay, combine_plots, combine_plots_nopp
from lcls.statistics import calculate_relative_p_values, generate_pp_lazy_data

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

import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

