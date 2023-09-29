

import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

def intensity_windowing(data, Imin, Imax):
    processed_data = np.copy(data)
    processed_data[processed_data < Imin] = 0
    processed_data[processed_data > Imax] = 0
    return processed_data

def define_rois(signal_center, background_center, shape):
    x_size, y_size = shape
    signal_roi = (slice(signal_center[0] - x_size // 2, signal_center[0] + x_size // 2),
                  slice(signal_center[1] - y_size // 2, signal_center[1] + y_size // 2))
    background_roi = (slice(background_center[0] - x_size // 2, background_center[0] + x_size // 2),
                      slice(background_center[1] - y_size // 2, background_center[1] + y_size // 2))
    return {"signal": signal_roi, "background": background_roi}

def bootstrap_resample(data):
    n_samples = data.shape[0]
    resample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return data[resample_indices, :, :]

def integrator(data, rois):
    signal_roi = rois["signal"]
    background_roi = rois["background"]
    signal_sum = np.sum(data[:, signal_roi[0], signal_roi[1]])
    background_sum = np.sum(data[:, background_roi[0], background_roi[1]])
    return signal_sum - background_sum

def empirical_std(data, rois, n_iterations=1000):
    values = [integrator(bootstrap_resample(data), rois) for _ in range(n_iterations)]
    return np.std(values, ddof=1)

def snr_study_with_constraints_updated(array, signal_center, background_center, max_size, min_separation, step_size):
    roi_sizes = []
    sb_values = []
    std_devs = []
    snrs = []
    for size in range(step_size, max_size + 1, step_size):
        separation = np.linalg.norm(np.array(signal_center) - np.array(background_center))
        if separation < 2*size + min_separation:
            continue
        rois = define_rois(signal_center, background_center, (size, size))
        sb_value = integrator(array, rois)
        std_dev = empirical_std(array, rois, n_iterations=1000)
        snr = sb_value / std_dev if std_dev != 0 else 0
        roi_sizes.append(size)
        sb_values.append(sb_value)
        std_devs.append(std_dev)
        snrs.append(snr)
    return roi_sizes, sb_values, std_devs, snrs
# analysis script
#import numpy as np
#import matplotlib.pyplot as plt
#
#def define_rois(signal_center, background_center, shape):
#    x_size, y_size = shape
#    signal_roi = (slice(signal_center[0] - x_size // 2, signal_center[0] + x_size // 2),
#                  slice(signal_center[1] - y_size // 2, signal_center[1] + y_size // 2))
#    background_roi = (slice(background_center[0] - x_size // 2, background_center[0] + x_size // 2),
#                      slice(background_center[1] - y_size // 2, background_center[1] + y_size // 2))
#    return {"signal": signal_roi, "background": background_roi}
#
#def snr_study_with_constraints_updated(array, signal_center, background_center, max_size, min_separation, step_size):
#    roi_sizes = []
#    sb_values = []
#    std_devs = []
#    snrs = []
#    for size in range(step_size, max_size + 1, step_size):
#        separation = np.linalg.norm(np.array(signal_center) - np.array(background_center))
#        if separation < 2*size + min_separation:
#            continue
#        rois = define_rois(signal_center, background_center, (size, size))
#        sb_value = integrator(array, rois)
#        std_dev = empirical_std(array, rois, n_iterations=1000)
#        snr = sb_value / std_dev if std_dev != 0 else 0
#        roi_sizes.append(size)
#        sb_values.append(sb_value)
#        std_devs.append(std_dev)
#        snrs.append(snr)
#    return roi_sizes, sb_values, std_devs, snrs
#
## Assuming the data is preprocessed and the functions `integrator` and `empirical_std` are loaded from the previous code
#new_com_signal = (88, 95)
#new_com_background_above = (new_com_signal[0] - 30, new_com_signal[1])
#
#roi_sizes_new_30, sb_values_new_30, std_devs_new_30, snrs_new_30 = snr_study_with_constraints_updated(
#    processed_data, new_com_signal, new_com_background_above, max_size=30, min_separation=10, step_size=1)
#
## Visualization
#plt.figure(figsize=(12, 8))
#plt.plot(roi_sizes_new_30, snrs_new_30, marker='o', label='SNR')
#plt.xlabel('ROI Size')
#plt.ylabel('SNR')
#plt.title('Signal to Noise Ratio for ROI sizes up to 30 (Using New ROI Centers)')
#plt.xlim(0, 30)
#plt.legend()
#plt.grid(True)
#plt.show()


# Unit tests
def test_all_functions():
    # Test for intensity_windowing
    test_array = np.array([[-5, 0, 5], [10, 15, 20]])
    processed = intensity_windowing(test_array, Imin=0, Imax=15)
    assert np.array_equal(processed, np.array([[0, 0, 5], [10, 15, 0]])), "Error in intensity_windowing."

    # Test for define_rois
    rois = define_rois((5, 5), (15, 15), (6, 6))
    assert rois["signal"] == (slice(2, 8), slice(2, 8)), "Error in define_rois - signal ROI."
    assert rois["background"] == (slice(12, 18), slice(12, 18)), "Error in define_rois - background ROI."

    # Test for bootstrap_resample
    test_array = np.ones((10, 5, 5))
    bootstrapped = bootstrap_resample(test_array)
    assert bootstrapped.shape == test_array.shape, "Error in bootstrap_resample."

    # Test for integrator
    test_array = np.ones((10, 20, 20))
    rois = {"signal": (slice(5, 10), slice(5, 10)), "background": (slice(10, 15), slice(10, 15))}
    result = integrator(test_array, rois)
    assert result == 0, "Error in integrator."

    # Test for empirical_std
    test_array = np.ones((10, 20, 20))
    rois = {"signal": (slice(5, 10), slice(5, 10)), "background": (slice(10, 15), slice(10, 15))}
    result = empirical_std(test_array, rois, n_iterations=50)
    assert result == 0, "Error in empirical_std."

    # Test for snr_study_with_constraints_updated
    test_array = np.ones((10, 20, 20))
    roi_sizes, _, _, _ = snr_study_with_constraints_updated(test_array, (5, 5), (15, 15), 5, 2, 1)
    assert roi_sizes == [1, 2, 3, 4, 5], "Error in snr_study_with_constraints_updated."

# End of code and unit tests

