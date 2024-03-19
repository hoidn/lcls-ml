from lcls.masks import compute_signal_mask, create_background_mask
from lcls.stacks import generate_plot_data

def run_analysis(cdw_output, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple, threshold,
                 data=None, histograms=None):
    """
    Runs the analysis pipeline for pump-probe data.

    :param cdw_output: Dictionary containing 'stacks_on' and 'stacks_off'.
    :param bin_boundaries: Boundaries for histogram bins.
    :param hist_start_bin: The starting bin for the histogram.
    :param roi_coordinates: ROI coordinates as a tuple (roi_x_start, roi_x_end, roi_y_start, roi_y_end).
    :param background_mask_multiple: Parameter for the plot_normalized_signal_vs_time_delay function.
    :param threshold: Threshold value for signal mask calculation.
    :param data: Optional data array for signal mask calculation.
    :param histograms: Optional histograms for signal mask calculation.
    :return: Dictionary capturing the outputs of the plot_normalized_signal_vs_time_delay function.
    """
    signal_mask = compute_signal_mask(bin_boundaries, hist_start_bin, roi_coordinates, threshold,
                                      data=data, histograms=histograms)
    background_mask = create_background_mask(signal_mask, background_mask_multiple, thickness=10)

    plot_results = generate_plot_data(cdw_output, signal_mask, background_mask, bin_boundaries, hist_start_bin,
                                      roi_coordinates)

    return plot_results
