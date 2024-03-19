import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from lcls.plots import geometric_mean

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
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(save_path)  # Save the figure to a file
    plt.show()  # Display the figure

def plot_normalized_signal_vs_time_delay(cdw_pp_output, signal_mask, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple):
    plot_data_dict = generate_plot_data(cdw_pp_output, signal_mask, bin_boundaries, hist_start_bin, roi_coordinates, background_mask_multiple)
    plot_data(plot_data_dict)
    return plot_data_dict

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
    plot_data(plot_data_dict1, subplot_spec=[gs[0, 1], gs[1, 1]], plot_title = 'Human')
    plot_data(cdw_data, subplot_spec=[gs[0, 0], gs[1, 0]], plot_title = 'Automated')
    plt.tight_layout()
    plt.show()

def combine_plots_nopp(cdw_data, human_data):
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    plot_data(cdw_data, subplot_spec=[gs[0, 0], gs[1, 0]], plot_title = 'Automated')
    plot_data(human_data, subplot_spec=[gs[0, 1], gs[1, 1]], plot_title = 'Human')
    plt.tight_layout()
    plt.show()
