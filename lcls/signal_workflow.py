import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import Label
import standard_workflow as helpers
import histogram_analysis
from pump_probe import optimize_signal_mask, CDW_PP
import pump_probe
from maskutils import erode_to_target, set_nearest_neighbors

def estimate_center(I0_x, I0_y):
    arg = (abs(I0_x)<2.)&(abs(I0_y)<3.)
    I0_x_mean,I0_y_mean = I0_x[arg].mean(),I0_y[arg].mean() # Mean position
    return I0_x_mean,I0_y_mean

parser = argparse.ArgumentParser(description="Signal-finding module for XPP experiments.")
parser.add_argument("run", type=int, help="Experiment run number")
parser.add_argument("exp", type=str, help="Experiment identifier")
parser.add_argument("h5dir", type=str, help="Directory for h5 files")
parser.add_argument("roi_crop", nargs=4, type=int, help="ROI crop coordinates")
parser.add_argument("roi_coordinates", nargs=4, type=int, help="ROI coordinates")
parser.add_argument("E0", type=float, help="X-ray energy")
parser.add_argument("--I0_thres", type=int, default=200, help="I0 monitor threshold")
parser.add_argument("--xc", type=float, default=-0.08, help="X-coordinate center")
parser.add_argument("--yc", type=float, default=-0.28, help="Y-coordinate center")
parser.add_argument("--xc_range", type=float, default=0.2, help="Range for xc filtering")
parser.add_argument("--yc_range", type=float, default=0.5, help="Range for yc filtering")
parser.add_argument("--min_peak_pixcount", type=int, default=1000, help="Minimum peak pixel count")
parser.add_argument("--Time_bin", type=float, default=2.0, help="Time bin width in picoseconds")
parser.add_argument("--TimeTool", nargs=2, type=float, default=[0, 0.005], help="TimeTool correction factors")
parser.add_argument("--Energy_Width", type=float, default=5, help="Energy width in eV")
parser.add_argument("--threshold_lower", type=float, default=0.0, help="Lower p value threshold for signal mask optimization")
parser.add_argument("--threshold_upper", type=float, default=0.3, help="Upper p value threshold for signal mask optimization")
parser.add_argument("--min_count", type=int, default=100, help="Minimum count for CDW_PP")

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
I0_thres = args.I0_thres
xc = args.xc
yc = args.yc
xc_range = args.xc_range
yc_range = args.yc_range
min_peak_pixcount = args.min_peak_pixcount
interpolate_gaps = args.interpolate_gaps
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

def combine_stacks(stacks: List[Dict[float, np.ndarray]]) -> np.ndarray:
    """
    Combines multiple stacks into a single 3D numpy array by concatenating the 3D arrays from each stack.

    Parameters:
        stacks (List[Dict[float, np.ndarray]]): A list of stacks, where each stack is a dictionary mapping time delays to 3D numpy arrays.

    Returns:
        np.ndarray: A single 3D numpy array obtained by stacking the 3D arrays from all provided stacks.
    """
    # Extract all 3D arrays from each stack and concatenate them
    combined_array = np.concatenate([array for stack in stacks for array in stack.values()], axis=0)

    return combined_array

imgs_thresh = combine_stacks([cdw_output['stacks_off']])

from lcls.histogram_analysis import calculate_histograms, run_histogram_analysis

data = imgs_thresh#[:7000, ...]#load_data(filepath)
histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)

if interpolate_gaps:
    set_nearest_neighbors(histograms, cdw_output['full_mask'], roi_crop)

compute_signal_mask = pump_probe.compute_signal_mask
calculate_signal_background_from_histograms = histogram_analysis.calculate_signal_background_from_histograms

best_signal_mask, best_params, grid_search_results = optimize_signal_mask(bin_boundaries, hist_start_bin, roi_coordinates, histograms,
                         threshold_lower=args.threshold_lower, threshold_upper=args.threshold_upper, num_threshold_points=15)

# TODO check if signal mask is valid
signal_mask = erode_to_target(best_signal_mask, min_peak_pixcount)
from matplotlib import pyplot as plt
import matplotlib.patches as patches
plt.imsave('signal_mask.png', signal_mask, cmap='gray')

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

## Plot 1: EMD Values
#axs[0, 0].imshow(res['emd_values'])
#axs[0, 0].set_title('EMD Values')
#
## Plot 2: 1 - Log(P Values)
#im = axs[0, 1].imshow(1 - np.log(res['p_values']), cmap='plasma')
#plt.colorbar(im, ax=axs[0, 1])
#axs[0, 1].set_title('1 - Log(P Values)')

# Plot 3: Heatmap with ROI
plot_heatmap_with_roi(axs[1, 0], histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end)
axs[1, 0].set_title('Heatmap with background ROI')

# Plot 4: Signal Mask
axs[1, 1].imshow(signal_mask)
axs[1, 1].set_title('Signal mask')

plt.tight_layout()
plt.show()
