
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.ndimage import label

import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, label

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


# New Functions

def sum_histograms_by_mask(histogram_array, binary_mask):
    # Apply the mask to select histograms
    masked_histograms = histogram_array[:, binary_mask]
    # Sum the selected histograms
    summed_histogram = np.mean(masked_histograms, axis=1)
    return summed_histogram

def melt_negative_clusters(cluster_array, N=1):
    inverted_array = np.logical_not(cluster_array)
    dilated_array = binary_dilation(inverted_array, iterations=N)
    return np.logical_or(cluster_array, np.logical_not(dilated_array))

def filter_negative_clusters_by_size(cluster_array, M=10):
    inverted_array = np.logical_not(cluster_array)
    labeled_array, num_features = label(inverted_array)
    cluster_sizes = np.bincount(labeled_array.ravel())
    small_clusters = np.where(cluster_sizes < M)[0]
    small_cluster_mask = np.isin(labeled_array, small_clusters)
    return np.logical_or(cluster_array, small_cluster_mask)

def visualize_clusters(cluster_array, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cluster_array, cmap='gray', origin='lower')
    plt.title(title)
    plt.xlabel('y-axis')
    plt.ylabel('x-axis')
    plt.show()

# Top-level functions for reproducing the visualizations

def visualize_size_filtered_clusters(histogram_array, binary_mask):
    summed_histogram = sum_histograms_by_mask(histogram_array, binary_mask)
    visualize_clusters(binary_mask, 'Size-Filtered Negative Clusters')

def visualize_histogram_comparison(histogram_array, binary_mask, bin_boundaries, hist_start_bin, save_path = None):
    energies = bin_boundaries[hist_start_bin + 1:]
    summed_histogram = sum_histograms_by_mask(histogram_array, binary_mask)
    summed_histogram_signal = sum_histograms_by_mask(histogram_array, ~binary_mask)
#     aggregate_histogram = np.sum(histogram_array, axis=(1, 2))
#     histogram_difference = aggregate_histogram - summed_histogram

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    axes[0].bar(energies, sum_histograms_by_mask(histogram_array, np.ones_like(binary_mask)), color='green', alpha=0.7)
    axes[0].set_title('Aggregate Histogram with No Filtering')
    axes[0].set_xlabel('energy (keV)')
    axes[0].set_ylabel('mean frequency')

    axes[1].bar(energies, summed_histogram, color='blue', alpha=0.7)
    axes[1].set_title('Aggregate histogram of non-signal pixels')
    axes[1].set_xlabel('energy (keV)')
    axes[1].set_ylabel('mean frequency')

    axes[2].bar(energies, summed_histogram_signal, color='red', alpha=0.7)
    axes[2].set_title('Aggregate histogram of signal candidate clusters')
    axes[2].set_xlabel('energy (keV)')
    axes[2].set_ylabel('mean frequency')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300)
    else:
        plt.show()

# Missing Functions

def run_histogram_analysis(data, bin_boundaries=np.arange(-10, 30, 0.2), hist_start_bin=60,
                           roi_x_start=30, roi_x_end=80, roi_y_start=40, roi_y_end=90, num_permutations=1000,
                          threshold = .1):
    histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)
    average_histogram = get_average_roi_histogram(histograms, roi_x_start, roi_x_end, roi_y_start, roi_y_end)
    emd_values = calculate_emd_values(histograms, average_histogram)
    
    null_distribution = generate_null_distribution(histograms, average_histogram, roi_x_start, roi_x_end, roi_y_start, roi_y_end,
                                                  num_permutations)
    p_values = calculate_p_values(emd_values, null_distribution)
    labeled_array, _ = identify_roi_connected_cluster(p_values, threshold, roi_x_start, roi_x_end, roi_y_start, roi_y_end)
    
    seed_x = (roi_x_start + roi_x_end) // 2
    seed_y = (roi_y_start + roi_y_end) // 2
    roi_cluster_label = labeled_array[seed_x, seed_y]
    roi_connected_cluster = (labeled_array == roi_cluster_label)
    
    return emd_values, p_values, labeled_array, roi_connected_cluster

def visualize_roi_connected_cluster(labeled_array, roi_x_start, roi_x_end, roi_y_start, roi_y_end):
    seed_x = (roi_x_start + roi_x_end) // 2
    seed_y = (roi_y_start + roi_y_end) // 2
    roi_cluster_label = labeled_array[seed_x, seed_y]
    roi_connected_cluster = (labeled_array == roi_cluster_label)
    
    visualize_clusters(roi_connected_cluster, 'Cluster Connected to the ROI')


