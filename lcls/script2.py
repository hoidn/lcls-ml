import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import Label
import standard_workflow as helpers
import histogram_analysis
from pump_probe import optimize_signal_mask
import pump_probe
from maskutils import erode_to_target, set_nearest_neighbors
from stacks import CDW_PP, extract_stacks_frames

def delay_bin(delay, delay_raw, Time_bin, arg_delay_nan):
    """
    """
    # TODO the bin values might be off by half a picosecond
    Time_bin = Time_bin * 1.0
    delay_min = np.floor(delay_raw[arg_delay_nan==False].min())
    delay_max = np.ceil(delay_raw[arg_delay_nan==False].max())
    bins = np.arange(delay_min, delay_max + Time_bin, Time_bin)
    binned_indices = np.digitize(delay, bins)
    binned_delays = np.array([bins[idx-1] if idx > 0 else bins[0] for idx in binned_indices])
#     print('Number of laser delays is: {0:d}, with an interval of {1:.2f} ps.'.format(num_delays,Time_bin))
    return binned_delays

def plot_data_bokeh(data, plot_title='Normalized Signal vs Time Delay'):
    output_file("plot.html")

    # Preparing data
    delays_on = data['delays_on']
    norm_signal_on = data['norm_signal_on']
    std_dev_on = data['std_dev_on']
    delays_off = data['delays_off']
    norm_signal_off = data['norm_signal_off']
    std_dev_off = data['std_dev_off']
    relative_p_values = data['relative_p_values']

    # Determine x-axis range
    x_min = min(min(delays_on), min(delays_off))
    x_max = max(max(delays_on), max(delays_off))

    # Plot 1 - Normalized Signal vs Time Delay modifications
    p1 = figure(title=plot_title, x_axis_label='Time Delay (ps)', y_axis_label='Normalized Signal', x_range=(x_min, x_max))
    p1.circle(delays_on, norm_signal_on, legend_label='Laser On: Signal', color="red")
    p1.line(delays_on, norm_signal_on, color="red", legend_label='Laser On: Connected')  # Added line
    p1.circle(delays_off, norm_signal_off, legend_label='Laser Off: Signal', color="black")
    p1.line(delays_off, norm_signal_off, color="black", legend_label='Laser Off: Connected')  # Added line

    # Enhanced error bars with increased thickness
    p1.segment(delays_on, norm_signal_on - std_dev_on, delays_on, norm_signal_on + std_dev_on, color="red", line_width=2)
    p1.segment(delays_off, norm_signal_off - std_dev_off, delays_off, norm_signal_off + std_dev_off, color="black", line_width=2)


    # Plot 2 - -log(P-value) vs Time Delay
    neg_log_p_values = [-np.log10(p) if p > 0 else 0 for p in relative_p_values]
    p2 = figure(title='-log(P-value) vs Time Delay', x_axis_label='Time Delay', y_axis_label='-log(P-value)', x_range=(x_min, x_max))
    p2.circle(sorted(set(delays_on) & set(delays_off)), neg_log_p_values, color="red", legend_label='-log(p-value)')

    # Threshold lines and labels
    for p_val, label in zip([0.5, 0.1, 0.01, 0.001], ['50%', '10%', '1%', '0.1%']):
        neg_log_p_val = -np.log10(p_val)
        p2.line([x_min, x_max], [neg_log_p_val, neg_log_p_val], line_dash="dashed", color="black")
        label = Label(x=x_max, y=neg_log_p_val, text=f'{label} level', y_offset=8)
        p2.add_layout(label)

    # Combine plots
    p = gridplot([[p1], [p2]])

    save(p)

def save_signal_mask_as_png(signal_mask, file_path='signal_mask.png', resolution=300):
    """
    """
    assert isinstance(signal_mask, (list, tuple, np.ndarray)), "signal_mask must be a 2D array-like structure"
    assert isinstance(file_path, str), "file_path must be a string"
    assert isinstance(resolution, int) and resolution > 0, "resolution must be a positive integer"

    plt.imshow(signal_mask)
    plt.title('Signal mask')
    plt.savefig(file_path, format='png', dpi=resolution)
    plt.close()  # Close the plot to free up memory


def estimate_center(I0_x, I0_y):
    arg = (abs(I0_x)<2.)&(abs(I0_y)<3.)
    I0_x_mean,I0_y_mean = I0_x[arg].mean(),I0_y[arg].mean() # Mean position
    return I0_x_mean,I0_y_mean

parser = argparse.ArgumentParser(description="Process X-ray data.")
parser.add_argument("run", type=int, help="Experiment run number")
parser.add_argument("exp", type=str, help="Experiment identifier")
parser.add_argument("h5dir", type=str, help="Directory for h5 files")
parser.add_argument("roi_crop", nargs=4, type=int, help="ROI crop coordinates")
parser.add_argument("roi_coordinates", nargs=4, type=int, help="ROI coordinates")
parser.add_argument("E0", type=float, help="X-ray energy")
parser.add_argument("--background_mask_multiple", type=float, default=1, help="Background mask multiple")
parser.add_argument("--separator_thickness", type=int, default=10, help="Separator thickness")
parser.add_argument("--I0_thres", type=int, default=200, help="I0 monitor threshold")
parser.add_argument("--xc", type=float, default=-0.08, help="X-coordinate center")
parser.add_argument("--yc", type=float, default=-0.28, help="Y-coordinate center")
parser.add_argument("--xc_range", type=float, default=0.2, help="Range for xc filtering")
parser.add_argument("--yc_range", type=float, default=0.5, help="Range for yc filtering")
parser.add_argument("--min_peak_pixcount", type=int, default=1000, help="Minimum peak pixel count")
parser.add_argument("--Time_bin", type=float, default=2.0, help="Time bin width in picoseconds")
parser.add_argument("--no_subtract_background", action="store_true", help="Do not subtract background from signal")
parser.add_argument("--TimeTool", nargs=2, type=float, default=[0, 0.005], help="TimeTool correction factors")
parser.add_argument("--Energy_Width", type=float, default=5, help="Energy width in eV")
parser.add_argument("--threshold_lower", type=float, default=0.0, help="Lower p value threshold for signal mask optimization")
parser.add_argument("--threshold_upper", type=float, default=0.3, help="Upper p value threshold for signal mask optimization")
parser.add_argument("--min_count", type=int, default=100, help="Minimum count for CDW_PP")

parser.add_argument("--estimate_center", action="store_true", help="Estimate the center coordinates xc and yc")
parser.add_argument("--interpolate_gaps", action="store_true", help="Interpolate detector gaps")
parser.add_argument("--delay_option", type=int, default=2, choices=[1, 2],
    help="Option for calculating xvar: 1 for lasDelay, 2 for lasDelay2 with FLTPOS_PS (default)")
parser.add_argument("--laser_delay_source", type=int, default=1, choices=[1, 2],
    help="Source of laser delay value: 1 for lasDelay (default), 2 for lasDelay2")
# TODO TimeTool = [0, 0.005]

args = parser.parse_args()

# Replace hardcoded values with args
run = args.run
exp = args.exp
h5dir = Path(args.h5dir)
roi_crop = args.roi_crop
roi_coordinates = args.roi_coordinates
roi_x_start, roi_x_end, roi_y_start, roi_y_end = roi_coordinates
E0 = args.E0
background_mask_multiple = helpers.background_mask_multiple = args.background_mask_multiple
separator_thickness = helpers.separator_thickness = args.separator_thickness
I0_thres = args.I0_thres
xc = args.xc
yc = args.yc
xc_range = args.xc_range
yc_range = args.yc_range
min_peak_pixcount = args.min_peak_pixcount
subtract_background = not args.no_subtract_background
interpolate_gaps = args.interpolate_gaps
estimate_center_flag = args.estimate_center
Time_bin = args.Time_bin
delay_option = args.delay_option
las_delay_source = args.laser_delay_source

min_count = args.min_count

TimeTool = args.TimeTool
Energy_Width = args.Energy_Width
Energy_Filter = [E0, Energy_Width]
IPM_pos_Filter = [xc_range, yc_range]

bin_boundaries = np.arange(5, 30, .2)
hist_start_bin = 1

cdw_output = CDW_PP(run, exp, h5dir, roi_crop, Energy_Filter, I0_thres, IPM_pos_Filter, Time_bin, TimeTool, las_delay_source, min_count=min_count)

from typing import List, Dict


imgs_thresh = extract_stacks_frames([cdw_output['stacks_off']])

from lcls.histogram_analysis import *

from scipy.stats import wasserstein_distance
from scipy.ndimage import label

data = imgs_thresh#[:7000, ...]#load_data(filepath)
histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)

if interpolate_gaps:
    set_nearest_neighbors(histograms, cdw_output['full_mask'], roi_crop)

threshold = .1
# Run the analysis
res = run_histogram_analysis(
    bin_boundaries = bin_boundaries, hist_start_bin = hist_start_bin,
    roi_x_start = roi_x_start, roi_x_end = roi_x_end, roi_y_start = roi_y_start,
    roi_y_end = roi_y_end, data = data,
    threshold = threshold)
signal_mask = res['signal_mask']

## Counting the number of True pixels in the signal mask
#true_pixels_count = np.sum(signal_mask)
#
## Test values for num_pixels
#num_pixels_exact = true_pixels_count
#num_pixels_double = 4 * true_pixels_count
#
## Running the create_continuous_buffer function with these test values
#continuous_buffer_exact = histogram_analysis.create_continuous_buffer(signal_mask, num_pixels = num_pixels_exact)
#continuous_buffer_double = histogram_analysis.create_continuous_buffer(signal_mask, num_pixels = num_pixels_double)
#
## Visualizing the results
#import matplotlib.pyplot as plt
#
#fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#
## Plotting the result with exact number of pixels
#axes[0].imshow(continuous_buffer_exact | signal_mask, cmap='gray')
#axes[0].set_title(f'BG mask (1x: {num_pixels_exact} pixels)')
#
## Plotting the result with double the number of pixels
#axes[1].imshow(continuous_buffer_double | signal_mask, cmap='gray')
#axes[1].set_title(f'BG mask (4x: {num_pixels_double} pixels)')
#
#plt.show()
#
#true_pixels_count, continuous_buffer_exact.shape, continuous_buffer_double.shape

compute_signal_mask = pump_probe.compute_signal_mask
calculate_signal_background_from_histograms = histogram_analysis.calculate_signal_background_from_histograms

best_signal_mask, best_background_mask, best_params, grid_search_results = optimize_signal_mask(
    bin_boundaries, hist_start_bin, roi_coordinates, histograms,
    threshold_lower=args.threshold_lower, threshold_upper=args.threshold_upper, 
    num_threshold_points=15, background_mask_multiple=args.background_mask_multiple, thickness=args.separator_thickness
)

signal_mask = erode_to_target(best_signal_mask, min_peak_pixcount)
background_mask = best_background_mask

cdw_data = pump_probe.generate_plot_data(
    cdw_output, signal_mask, background_mask, bin_boundaries,
    hist_start_bin, roi_coordinates, subtract_background=subtract_background
)
pump_probe.plot_data(cdw_data)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# TODO cleanup the code below. move the function to the top of the module, where other functions are
# background_mask = histogram_analysis.create_background_mask(auto_signal_mask, background_mask_multiple, 1.,
#                                                             separator_thickness=40)

def plot_heatmap_with_roi(ax, data, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    # Integrate data over the 0th axis
    integrated_data = data.mean(axis=0)

    # Create the heatmap
    im = ax.imshow(integrated_data, cmap='hot', interpolation='nearest',
                  vmin = np.percentile(integrated_data.ravel(), 10))

    # Create a Rectangle patch
    rect = patches.Rectangle((roi_y_start, roi_x_start), roi_y_end - roi_y_start, roi_x_end - roi_x_start,
                             linewidth=5, edgecolor='blue', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
#     plt.colorbar(im, ax = ax)

# Create a multi-panel plot in a 2x2 layout
fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjusted for 2x2 layout

# Plot 1: EMD Values
axs[0, 0].imshow(res['emd_values'])
axs[0, 0].set_title('EMD Values')

# Plot 2: 1 - Log(P Values)
im = axs[0, 1].imshow(1 - np.log(res['p_values']), cmap='plasma')
plt.colorbar(im, ax=axs[0, 1])
axs[0, 1].set_title('1 - Log(P Values)')

# Plot 3: Heatmap with ROI
plot_heatmap_with_roi(axs[1, 0], histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end)
axs[1, 0].set_title('Heatmap with background ROI')

# Plot 4: Signal Mask
axs[1, 1].imshow(signal_mask)
axs[1, 1].set_title('Signal mask')

plt.tight_layout()
plt.show()

