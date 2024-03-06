import numpy as np

def delay_bin(delay, delay_raw, Time_bin, arg_delay_nan):
    print("Starting delay binning...")
    # Adjust the bin width to ensure it's a float
    Time_bin = float(Time_bin)

    # Determine the minimum and maximum values from the non-NaN delays
    delay_min = np.floor(delay_raw[arg_delay_nan == False].min())
    delay_max = np.ceil(delay_raw[arg_delay_nan == False].max())
    print(f"Delay min: {delay_min}, Delay max: {delay_max}")

    # Create bins that are shifted by half the bin width
    half_bin = Time_bin / 2
    bins = np.arange(delay_min - half_bin, delay_max + Time_bin, Time_bin)
    print(f"Generated {len(bins)} bins with bin width {Time_bin}.")

    # Assign each delay to the nearest bin
    binned_indices = np.digitize(delay, bins, right=True)
    print(f"Binned indices range: {binned_indices.min()} to {binned_indices.max()}")

    # Convert bin indices to delay values
    binned_delays = bins[binned_indices - 1] + half_bin
    print(f"Binned delays range: {binned_delays.min()} to {binned_delays.max()}")

    # Ensure that the binned delays are within the min and max range
    binned_delays = np.clip(binned_delays, delay_min, delay_max)
    print(f"Final binned delays range after clipping: {binned_delays.min()} to {binned_delays.max()}")

    print(f"Generated {len(np.unique(binned_delays))} unique binned delays.")
    return binned_delays

def extract_stacks_by_delay(binned_delays, img_array, bin_width, min_count, ROI_mask, arg_laser, I0):
    print("Unique binned delays:", np.unique(binned_delays))
    unique_binned_delays = np.unique(binned_delays)
    stacks = {}

    for d in unique_binned_delays:
        specific_mask = (binned_delays == d) & arg_laser
        stack = img_array[specific_mask]
        stack_I0 = I0[specific_mask]
        stack_binned_delays = binned_delays[specific_mask]

        if stack.shape[0] >= min_count:
            assert stack.shape[0] == stack_binned_delays.shape[0] == stack_I0.shape[0], "Inconsistent shapes in stack arrays"
            stacks[d] = {
                'images': stack,
                'delays': stack_binned_delays,
                'I0': stack_I0,
            }
        else:
            print(f"Dropped delay {d} due to count {stack.shape[0]} being less than minimum required {min_count}.")

    return stacks

from typing import List, Dict
def combine_stacks(stacks: List[Dict[str, Dict[str, np.ndarray]]]) -> np.ndarray:
    """
    Combines the 'images' arrays from a list of dictionaries of dictionaries into a single 3D numpy array by concatenating them. Each inner dictionary is expected to contain an 'images' key mapping to a 3D numpy array.

    Parameters:
        stacks (List[Dict[str, Dict[str, np.ndarray]]]): A list of dictionaries, where each dictionary represents a condition with keys mapping to another dictionary containing 'images', 'delays', and 'I0'.

    Returns:
        np.ndarray: A single 3D numpy array obtained by concatenating the 'images' arrays from all provided stacks.
    """
    # Extract 'images' arrays from each stack and concatenate them
    combined_images = []
    for stack in stacks:
        for condition, data in stack.items():
            combined_images.append(data['images'])
    combined_array = np.concatenate(combined_images, axis=0)

    return combined_array

def CDW_PP(Run_Number, exp, h5dir, ROI, Energy_Filter, I0_Threshold, IPM_pos_Filter, Time_bin, TimeTool, las_delay_source,
          min_count = 200):
    from smd import SMD_Loader, EnergyFilter
    rr = SMD_Loader(Run_Number, exp, h5dir)  # Small Data Import

    # Mask for bad pixels
    idx_tile = rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][0,0]
    full_mask = rr.UserDataCfg.jungfrau1M.mask[idx_tile][rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][1,1],rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,0]:rr.UserDataCfg.jungfrau1M.ROI_0__ROI_0_ROI[()][2,1]]
    ROI_mask = full_mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]

    I0 = rr.ipm2.sum[:]
    arg_I0 = (I0 >= I0_Threshold)

    # IPM Positional Filter
    I0_x = rr.ipm2.xpos[:]
    I0_y = rr.ipm2.ypos[:]
    arg = (abs(I0_x) < 2.) & (abs(I0_y) < 3.)
    I0_x_mean, I0_y_mean = I0_x[arg].mean(), I0_y[arg].mean()
    arg_I0_x = (I0_x < (I0_x_mean + IPM_pos_Filter[0])) & (I0_x > (I0_x_mean - IPM_pos_Filter[0]))
    arg_I0_y = (I0_y < (I0_y_mean + IPM_pos_Filter[1])) & (I0_y > (I0_y_mean - IPM_pos_Filter[1]))
    

    # Time Tool Logic
    tt_arg = TimeTool[0]
    delay_source = rr.enc.lasDelay if las_delay_source == 1 else rr.enc.lasDelay2
    delay = np.array(delay_source) + np.array(rr.tt.FLTPOS_PS) * tt_arg
    arg_delay_nan = np.isnan(delay)

    # Energy Filtering
    imgs = EnergyFilter(rr, Energy_Filter, ROI)

    # Laser On/Off Logic
    arg_laser_on = (np.array(rr.evr.code_90) == 1.) #& (( arg_I0_x & arg_I0_y ) & arg_I0)
    arg_laser_off = (np.array(rr.evr.code_91) == 1.)# & (( arg_I0_x & arg_I0_y ) & arg_I0)

    binned_delays = delay_bin(delay, np.array(delay_source), Time_bin, arg_delay_nan)

    stacks_on = extract_stacks_by_delay(binned_delays, imgs, Time_bin, min_count, ROI_mask, arg_laser_on, I0)
    stacks_off = extract_stacks_by_delay(binned_delays, imgs, Time_bin, min_count, ROI_mask, arg_laser_off, I0)

    return {
    'stacks_on': stacks_on,
    'stacks_off': stacks_off,
    'I0': I0,
    'binned_delays': binned_delays,
    'arg_laser_on': arg_laser_on,
    'arg_laser_off': arg_laser_off,
    'full_mask': full_mask,
    'roi_mask': ROI_mask
    }
