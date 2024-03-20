import numpy as np
from scipy.stats import norm

def calculate_relative_p_values(Intensity_on, Intensity_off, assume_photon_counts=True):
    p_values = []
    for i in range(len(Intensity_on)):
        signal_on = Intensity_on[i, 0]
        signal_off = Intensity_off[i, 0]
        if assume_photon_counts:
            std_dev_on = np.sqrt(signal_on)
            std_dev_off = np.sqrt(signal_off)
        else:
            std_dev_on = Intensity_on[i, 1]
            std_dev_off = Intensity_off[i, 1]
        delta_signal = abs(signal_on - signal_off)
        combined_std_dev = np.sqrt(std_dev_on**2 + std_dev_off**2)
        z_score = delta_signal / combined_std_dev
        p_value = 2 * (1 - norm.cdf(z_score))  # Two-tailed test
        p_values.append(p_value)
    return np.array(p_values)

def generate_pp_lazy_data(imgs_on, imgs_off, mask, delay, assume_photon_counts=False):
    Intensity_on, Intensity_off = [], []
    npixels = (mask == 1).sum()
    for i in range(imgs_on.shape[0]):
        Intensity_on.append(imgs_on[i][mask == 1].mean())
        Intensity_on.append(imgs_on[i][mask == 1].std() / np.sqrt(npixels))
        Intensity_off.append(imgs_off[i][mask == 1].mean())
        Intensity_off.append(imgs_off[i][mask == 1].std() / np.sqrt(npixels))
    Intensity_on = np.array(Intensity_on).reshape(imgs_on.shape[0], 2)
    Intensity_off = np.array(Intensity_off).reshape(imgs_on.shape[0], 2)
    p_values = calculate_relative_p_values(Intensity_on, Intensity_off, assume_photon_counts)
    return {
        'delay': delay,
        'Intensity_on': Intensity_on,
        'Intensity_off': Intensity_off,
        'p_values': p_values
    }

def calculate_figure_of_merit(analysis_results):
    """
    Calculates the geometric mean of the p-values from the analysis results.

    :param analysis_results: Dictionary containing analysis results with p-values.
    :return: Geometric mean of the p-values.
    """
    p_values = analysis_results['relative_p_values']  # Extract p-values from the results
    p_values = np.array(p_values)
    p_values = p_values[p_values > 0]  # Exclude non-positive values for geometric mean calculation
    if len(p_values) == 0:
        return 0  # Return 0 if there are no positive p-values
    geometric_mean = np.exp(np.mean(np.log(p_values)))
    return 1 - geometric_mean
