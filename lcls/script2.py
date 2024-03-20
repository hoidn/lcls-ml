#from bokeh.layouts import gridplot
#from bokeh.models import Label
#from bokeh.plotting import figure, output_file, save
from maskutils import erode_to_target, set_nearest_neighbors
from pathlib import Path
from pump_probe import optimize_signal_mask
from scipy.ndimage import label
from scipy.stats import wasserstein_distance
from stacks import CDW_PP, extract_stacks_frames
from typing import List, Dict
import argparse
import histogram_analysis
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pump_probe
#import standard_workflow as helpers

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
#background_mask_multiple = helpers.background_mask_multiple = args.background_mask_multiple
#separator_thickness = helpers.separator_thickness = args.separator_thickness
background_mask_multiple = args.background_mask_multiple
separator_thickness = args.separator_thickness
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


imgs_thresh = extract_stacks_frames([cdw_output['stacks_off']])

from histogram_analysis import *


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

compute_signal_mask = pump_probe.compute_signal_mask
calculate_signal_background_from_histograms = histogram_analysis.calculate_signal_background_from_histograms

best_signal_mask, best_background_mask, best_params, grid_search_results = optimize_signal_mask(
    bin_boundaries, hist_start_bin, roi_coordinates, histograms,
    threshold_lower=args.threshold_lower, threshold_upper=args.threshold_upper, 
    num_threshold_points=15
)

signal_mask = erode_to_target(best_signal_mask, min_peak_pixcount)
background_mask = best_background_mask

cdw_data = pump_probe.generate_plot_data(
    cdw_output, signal_mask, background_mask, bin_boundaries,
    hist_start_bin, roi_coordinates, subtract_background=subtract_background
)
pump_probe.plot_data(cdw_data)


# background_mask = histogram_analysis.create_background_mask(auto_signal_mask, background_mask_multiple, 1.,
#                                                             separator_thickness=40)

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

