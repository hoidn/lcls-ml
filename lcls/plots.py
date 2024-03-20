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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_data(data, subplot_spec=None, plot_title='Normalized Signal vs Time Delay', save_path='plot.png'):
    """
    Plots the normalized signal vs time delay data.

    :param data: Dictionary containing the data to plot.
    :param subplot_spec: Optional subplot specification for embedding the plot in a larger figure.
    :param plot_title: Title for the plot.
    :param save_path: Path to save the plot image.
    """
    fig = plt.gcf()

    if subplot_spec is None:
        gs = GridSpec(2, 1)  # Default to 2x1 grid
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
    ax1.errorbar(delays_off, norm_signal_off, yerr=std_dev_off, fmt='ks-', mec='k', mfc='white', alpha=0.2,
                 label='Laser Off: Signal')
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
        ax2.text(ax2.get_xlim()[1], neg_log_p_val + label_offset, f'{label} level', va='center', ha='right',
                 fontsize=18, color='black')

    ax2.legend()
    ax2.set_title('FOM: {:.2f}'.format(-np.log10(geometric_mean(relative_p_values))))

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(save_path)  # Save the figure to a file
    plt.show()  # Display the figure

