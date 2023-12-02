import matplotlib.pyplot as plt
import numpy as np

import pvalues
import histogram_analysis

compute_aggregate_pvals = pvalues.compute_aggregate_pvals
calculate_signal_background_noI0 = histogram_analysis.calculate_signal_background_noI0
calculate_histograms = histogram_analysis.calculate_histograms

compute_aggregate_pvals_with_custom_background = pvalues.compute_aggregate_pvals_with_custom_background


from typing import Dict, List, Tuple, Any

def geometric_mean(arr):
    """
    Calculate the geometric mean of a numpy array.

    :param arr: numpy array
    :return: geometric mean of the array
    """
    return np.exp(np.mean(np.log(arr)))

from pump_probe import process_stacks

