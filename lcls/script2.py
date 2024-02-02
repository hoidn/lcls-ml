import argparse
import numpy as np
import tables
import matplotlib.pyplot as plt
from pathlib import Path
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import Label
import standard_workflow as helpers
import histogram_analysis
from standard_workflow import generate_intensity_data
from maskutils import generate_mask, erode_to_target, set_nearest_neighbors
from pump_probe import optimize_signal_mask, CDW_PP
import pump_probe

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

def SMD_Loader(Run_Number, exp, h5dir):
    # Load the Small Data
    fname = '{}_Run{:04d}.h5'.format(exp,Run_Number)
    fname = h5dir / fname
    rr = tables.open_file(fname).root # Small Data
    print(fname)
    return rr

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

parser.add_argument("--estimate_center", action="store_true", help="Estimate the center coordinates xc and yc")
parser.add_argument("--interpolate_gaps", action="store_true", help="Interpolate gaps in the data")
parser.add_argument("--delay_option", type=int, default=2, choices=[1, 2],
    help="Option for calculating xvar: 1 for lasDelay, 2 for lasDelay2 with FLTPOS_PS (default)")
# TODO TimeTool = [0, 0.005]

args = parser.parse_args()

# Replace hardcoded values with args
run = args.run
exp = args.exp
h5dir = Path(args.h5dir)
roi_crop = args.roi_crop
roi_coordinates = args.roi_coordinates
E0 = args.E0
background_mask_multiple = helpers.background_mask_multiple = args.background_mask_multiple
separator_thickness = helpers.separator_thickness = args.separator_thickness
I0_thres = args.I0_thres
xc = args.xc
yc = args.yc
xc_range = args.xc_range
yc_range = args.yc_range
min_peak_pixcount = args.min_peak_pixcount
interpolate_gaps = args.interpolate_gaps
estimate_center_flag = args.estimate_center
Time_bin = args.Time_bin
delay_option = args.delay_option

TimeTool = [0, 0.005]
Energy_Filter = [E0, 5]

rr = SMD_Loader(run, exp, h5dir)
rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()] # ROI used for generating the Small Data
idx_tile = rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0,0]
print(rr.jungfrau1M.ROI_0_area.shape)

imgs_thresh = rr.jungfrau1M.ROI_0_area[:,roi_crop[0]:roi_crop[1],roi_crop[2]:roi_crop[3]]
imgs_thresh[imgs_thresh<4.] = 0

idx_on = np.where(np.array(rr.evr.code_90)==1.)[0]
idx_off = np.where(np.array(rr.evr.code_91)==1.)[0]
idx_on.shape,idx_off.shape

if delay_option == 1:
    tt_arg = 0# TODO
    xvar = rr.enc.lasDelay
    xvar = np.array(rr.enc.lasDelay) + np.array(rr.tt.FLTPOS_PS)*tt_arg # in picosecond
    arg_delay_nan = np.isnan(xvar) # Some events report NaN
    xvar_unique = np.array(sorted(list(set(delay_bin(xvar,np.array(rr.enc.lasDelay),Time_bin,arg_delay_nan))))) # Time binning

else:
    xvar = rr.enc.lasDelay2 + np.array(rr.tt.FLTPOS_PS)*0.
    xvar = np.round(xvar)

    xvar_unique = np.array(list(set(xvar)))
    idx_nan = np.where(np.isnan(xvar_unique)==1.)
    xvar_unique = np.delete(xvar_unique,idx_nan)
    xvar_unique.shape,xvar_unique

    xvar_unique.sort()
    xvar_unique,xvar_unique.shape

    xvar_unique = np.linspace(xvar_unique.min(),xvar_unique.max(),33)
    print(xvar_unique)


    xvar=rr.enc.lasDelay2 + np.array(rr.tt.FLTPOS_PS)*0
for i in range(len(xvar)):
    if np.isnan(xvar[i])==True:
        continue
    diff = abs(xvar[i]-xvar_unique)
    idx = np.where(diff==diff.min())[0][0]
    xvar[i] = xvar_unique[idx]

I0_a = rr.ipm2.sum[:]
I0_x = rr.ipm2.xpos[:]
I0_y = rr.ipm2.ypos[:]
if estimate_center_flag:
    xc, yc = estimate_center(I0_x, I0_y)

arg = (I0_x<(xc+xc_range))&(I0_x>(xc-xc_range))&(I0_y<(yc+yc_range))&(I0_y>(yc-yc_range))

# TODO this has to be set manually
mask = rr.UserDataCfg.jungfrau1M.mask[idx_tile][rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,1],rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,1]]

im = imgs_thresh[(I0_a>I0_thres)&(np.array(rr.evr.code_90)==1.),:,:].mean(axis=0)
im = im*mask[roi_crop[0]:roi_crop[1],roi_crop[2]:roi_crop[3]]

im1 = imgs_thresh[(I0_a>I0_thres)&(np.array(rr.evr.code_91)==1.),:,:].mean(axis=0)
im1 = im1*mask[roi_crop[0]:roi_crop[1],roi_crop[2]:roi_crop[3]]

#roi = [0,im.shape[0]-30,0,im.shape[1]-30]
#cdw_mask = np.zeros_like(im)
#cdw_mask[roi[0]:roi[1],roi[2]:roi[3]] = 1.
#print(roi)

ims_crop = imgs_thresh
I = ims_crop.mean(axis=(1,2))


# Redefine bin boundaries
bin_boundaries = np.arange(5, 30, .2)
hist_start_bin = 1

data = imgs_thresh#[:7000, ...]#load_data(filepath)
histograms = histogram_analysis.calculate_histograms(data, bin_boundaries, hist_start_bin)

tmp = histograms.copy()
histograms = tmp.copy()

# TODO
#if interpolate_gaps:
#    set_nearest_neighbors(histograms, mask, roi_crop)

plt.imshow(histograms.sum(axis = 0))
plt.colorbar()

# TODO use just laser off histograms
signal_mask, best_params, grid_search_results = optimize_signal_mask(bin_boundaries, hist_start_bin,
                        roi_coordinates, histograms,
                        threshold_lower=0.0, threshold_upper=.3, num_threshold_points=15)

auto_signal_mask = erode_to_target(signal_mask, min_peak_pixcount)
save_signal_mask_as_png(auto_signal_mask)

np.sqrt(signal_mask.sum())
# plt.imshow(auto_signal_mask)

cdw_output = CDW_PP(run, roi_crop,
                    Energy_Filter, I0_thres,
                    [xc_range, yc_range], Time_bin, TimeTool)

cdw_data = pump_probe.generate_plot_data(cdw_output, signal_mask, bin_boundaries,
                              hist_start_bin, roi_coordinates, background_mask_multiple)
#cdw_output = generate_intensity_data(auto_signal_mask, xvar_unique, arg, I, xvar, I0_a, I0_thres, ims_crop, rr, background_mask_multiple)
plot_data_bokeh(cdw_data)