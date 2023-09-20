
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.ndimage import label

# def load_data(filepath):
#     """Load the data from the given file."""
#     return imgs_thresh

# def calculate_histograms(data, bin_boundaries, hist_start_bin):
#     """Generate histograms for the data."""
#     histograms = np.apply_along_axis(lambda x: np.histogram(x, bins=bin_boundaries)[0], 0, data)
#     histograms = histograms + 1e-9
#     histograms = histograms / (1e-9 + np.sum(histograms, axis=0))
#     return histograms[hist_start_bin:, :, :]

def calculate_histograms(data, bin_boundaries, hist_start_bin):
    """Generate histograms for the data."""
    expected_shape = (len(bin_boundaries) - 1, data.shape[1], data.shape[2])
    histograms = np.zeros(expected_shape)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            histogram, _ = np.histogram(data[:, i, j], bins=bin_boundaries)
            histograms[:len(histogram), i, j] = histogram
    histograms = histograms + 1e-9
    normalized_histograms = histograms / (1e-9 + np.sum(histograms, axis=0))
    return normalized_histograms[hist_start_bin:, :, :]


def get_average_roi_histogram(histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    """Calculate the average histogram for the ROI."""
    roi_histograms = histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    average_roi_histogram = np.mean(roi_histograms, axis=(1, 2))
    return average_roi_histogram / np.sum(average_roi_histogram)

def calculate_emd_values(histograms, average_histogram):
    """Compute the Earth Mover's Distance for each histogram."""
    emd_values = np.zeros((histograms.shape[1], histograms.shape[2]))
    for i in range(histograms.shape[1]):
        for j in range(histograms.shape[2]):
            emd_values[i, j] = wasserstein_distance(histograms[:, i, j], average_histogram)
    return emd_values

def generate_null_distribution(histograms, average_histogram, roi_x_start, roi_x_end, roi_y_start, roi_y_end, num_permutations=1000):
    """Create the null distribution using histograms within the ROI."""
    null_emd_values = []
    for _ in range(num_permutations):
        random_x_idx = np.random.choice(range(roi_x_start, roi_x_end))
        random_y_idx = np.random.choice(range(roi_y_start, roi_y_end))
        shuffled_histogram = histograms[:, random_x_idx, random_y_idx]
        null_emd_values.append(wasserstein_distance(shuffled_histogram, average_histogram))
    return np.array(null_emd_values)

# def calculate_p_values(emd_values, null_distribution):
#     """Compute p-values based on the observed EMD values and the null distribution."""
#     p_values = np.zeros_like(emd_values)
#     for i in range(emd_values.shape[0]):
#         for j in range(emd_values.shape[1]):
#             p_values[i, j] = np.mean(emd_values[i, j] >= null_distribution)
#     return p_values

def calculate_p_values(emd_values, null_distribution):
    """Vectorized computation of p-values based on the observed EMD values and the null distribution."""
    return np.mean(emd_values[:, :, np.newaxis] >= null_distribution, axis=2)

def identify_roi_connected_cluster(p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    """Find the cluster connected to the ROI."""
    porous_pixels = p_values > threshold
    labeled_array, _ = label(porous_pixels)
    seed_x = (roi_x_start + roi_x_end) // 2
    seed_y = (roi_y_start + roi_y_end) // 2
    roi_cluster_label = labeled_array[seed_x, seed_y]
    return labeled_array, labeled_array == roi_cluster_label
    #return labeled_array == roi_cluster_label
