from deps import calculate_p_value
import histogram_analysis
import numpy as np

separator_thickness=10
background_mask_multiple = 1

@histogram_analysis.memoize_general
def calculate_mean_over_0th_axis(images, I0):
    normalized_images = images / I0[:, np.newaxis, np.newaxis]
    return normalized_images.mean(axis=0)

@histogram_analysis.memoize_general
def calculate_std_over_0th_axis(images, I0):
    normalized_images = images / I0[:, np.newaxis, np.newaxis]
    return normalized_images.std(axis=0)

def process_data(signal_mask, xvar_unique, arg, I, xvar, I0_a, I0_thres, ims_crop, rr):
    ims_group, ims_group_off, ims_group_std, ims_group_off_std, scan_motor = [], [], [], [], []
    print('xvar unique in process_data', xvar_unique)
    for xvar_val in xvar_unique:
        idx = np.where(arg & (I > 0.0) & (xvar == xvar_val) & (I0_a > I0_thres) & (I0_a < 15000) & (np.array(rr.evr.code_90) == 1))
        idx_off = np.where(arg & (I > 0.0) & (xvar == xvar_val) & (I0_a > I0_thres) & (I0_a < 15000) & (np.array(rr.evr.code_91) == 1))

        if len(idx[0]) > 0 and len(idx_off[0]) > 0:
            mean_on_0th = calculate_mean_over_0th_axis(ims_crop[idx], I0_a[idx])
            std_on_0th = calculate_std_over_0th_axis(ims_crop[idx], I0_a[idx])
            mean_off_0th = calculate_mean_over_0th_axis(ims_crop[idx_off], I0_a[idx_off])
            std_off_0th = calculate_std_over_0th_axis(ims_crop[idx_off], I0_a[idx_off])

            mean_on = np.mean(mean_on_0th[signal_mask])
            std_on = std_on_0th[signal_mask]
            mean_off = np.mean(mean_off_0th[signal_mask])
            std_off = std_off_0th[signal_mask]

            ims_group.append(mean_on)
            ims_group_std.append(std_on)
            ims_group_off.append(mean_off)
            ims_group_off_std.append(std_off)
            scan_motor.append(xvar_val)

    Intensity = np.array([np.mean(group) for group in ims_group])
    ssum = signal_mask.sum()
    std = np.array([np.sqrt(np.sum(group**2)) / np.sqrt(ssum * len(idx[0])) for group in ims_group_std])  # Sum in quadrature for standard deviation
    bg = np.array([np.mean(group) for group in ims_group_off])

    print('scan_motor in process_data', xvar_unique)
    return scan_motor, Intensity, std, bg

def calculate_intensity_differences(auto_signal_mask, xvar_unique, arg, I, xvar, I0_a, I0_thres, ims_crop, rr, background_mask_multiple):
    from histogram_analysis import create_background_mask

    # Create background mask
    background_mask = create_background_mask(auto_signal_mask, background_mask_multiple, 1., separator_thickness=separator_thickness)

    # Process data for signal
    delays, intensity_on_signal, std_signal, intensity_off_signal = process_data(auto_signal_mask, xvar_unique, arg, I, xvar, I0_a, I0_thres, ims_crop, rr)

    # Process data for background
    _, intensity_on_background, std_background, intensity_off_background = process_data(background_mask, xvar_unique, arg, I, xvar, I0_a, I0_thres, ims_crop, rr)

    # Calculate intensity differences and standard deviation in quadrature
    intensity_difference_on = intensity_on_signal - intensity_on_background
    intensity_difference_off = intensity_off_signal - intensity_off_background
    std_quadrature = (std_signal**2 + std_background**2)**0.5

    return delays, intensity_difference_on, intensity_difference_off, std_quadrature

def generate_intensity_data(auto_signal_mask, xvar_unique, arg, I, xvar, I0_a, I0_thres, ims_crop, rr, background_mask_multiple):
    # Presuming the existence of calculate_intensity_differences, normalize_signal, and calculate_p_value functions
    # Calculate intensity differences
    delays, intensity_diff_on, intensity_diff_off, std_quadrature = calculate_intensity_differences(auto_signal_mask, xvar_unique, arg, I, xvar, I0_a, I0_thres, ims_crop, rr, background_mask_multiple)
    print('top level delays', delays)

    # Normalize the signals
    norm_signal_on = intensity_diff_on
    norm_signal_off = intensity_diff_off

    # Calculate relative p-values
    relative_p_values = []
    for delay in delays:
        signal_on = norm_signal_on[delays.index(delay)]
        signal_off = norm_signal_off[delays.index(delay)]
        std_dev_on_val = std_quadrature[delays.index(delay)]  # Assuming same std deviation for 'on' and 'off'
        std_dev_off_val = std_dev_on_val  # As above

        p_value = calculate_p_value(signal_on, signal_off, std_dev_on_val, std_dev_off_val)
        relative_p_values.append(p_value)

    return {
        'delays_on': delays,
        'norm_signal_on': norm_signal_on,
        'std_dev_on': std_quadrature,
        'delays_off': delays,
        'norm_signal_off': norm_signal_off,
        'std_dev_off': std_quadrature,
        'relative_p_values': relative_p_values
    }

