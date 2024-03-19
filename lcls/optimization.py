from lcls.analysis import run_analysis
from lcls.masks import compute_signal_mask
from lcls.plots import calculate_figure_of_merit

def optimize_analysis(cdw_output, bin_boundaries, hist_start_bin, roi_coordinates, param_grid, histograms=None):
    """
    Optimizes the analysis parameters using grid search.

    :param cdw_output: Dictionary containing 'stacks_on' and 'stacks_off'.
    :param bin_boundaries: Boundaries for histogram bins.
    :param hist_start_bin: The starting bin for the histogram.
    :param roi_coordinates: ROI coordinates as a tuple (roi_x_start, roi_x_end, roi_y_start, roi_y_end).
    :param param_grid: Dictionary defining the grid search space for 'background_mask_multiple' and 'threshold'.
    :param histograms: Optional histograms for signal mask calculation.
    :return: Dictionary containing the optimal parameters and the corresponding figure of merit.
    """
    optimal_params = None
    best_figure_of_merit = float('-inf')

    for background_mask_multiple in param_grid['background_mask_multiple']:
        for threshold in param_grid['threshold']:
            analysis_results = run_analysis(cdw_output, bin_boundaries, hist_start_bin, roi_coordinates,
                                            background_mask_multiple, threshold, histograms=histograms)

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

def optimize_signal_mask(bin_boundaries, hist_start_bin, roi_coordinates, histograms,
                         threshold_lower=0., threshold_upper=.5, num_threshold_points=10, num_runs=5,
                         max_signal_fraction=0.4):
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
            signal_mask = compute_signal_mask(bin_boundaries, hist_start_bin, roi_coordinates, threshold,
                                              histograms=histograms)
            if signal_mask.sum() == 0 or signal_mask.mean() > max_signal_fraction:
                continue

            background_mask = create_background_mask(signal_mask, 1, 10)
            signal, bg, _ = hist.calculate_signal_background_from_histograms(histograms, signal_mask,
                                                                             background_mask, bin_boundaries,
                                                                             hist_start_bin)

            if bg == 0:
                continue

            ratio = (signal - bg) / bg
            ratios.append(ratio)

        if not ratios:
            continue

        avg_ratio = np.mean(ratios)
        std_dev = np.std(ratios)
        grid_search_results.append((threshold, avg_ratio, std_dev))

        if avg_ratio > best_avg_ratio:
            best_avg_ratio = avg_ratio
            best_signal_mask = signal_mask
            best_threshold = threshold

    grid_search_results_np = np.array(grid_search_results,
                                      dtype=[('threshold', float), ('avg_ratio', float), ('std_dev', float)])

    return best_signal_mask, best_threshold, grid_search_results_np
