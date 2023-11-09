
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

def calculate_histograms(data, bin_boundaries, hist_start_bin):
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
    roi_histograms = histograms[:, roi_x_start:roi_x_end, roi_y_start:roi_y_end]
    average_roi_histogram = np.mean(roi_histograms, axis=(1, 2))
    return average_roi_histogram / np.sum(average_roi_histogram)

def calculate_emd_values(histograms, average_histogram):
    from scipy.stats import wasserstein_distance
    emd_values = np.zeros((histograms.shape[1], histograms.shape[2]))
    for i in range(histograms.shape[1]):
        for j in range(histograms.shape[2]):
            emd_values[i, j] = wasserstein_distance(histograms[:, i, j], average_histogram)
    return emd_values

def identify_background_pixels(cluster_image, neighbor_threshold):
    image_height, image_width = cluster_image.shape
    neighborhood_window = np.array([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],           [0, 1],
        [1, -1], [1, 0], [1, 1]
    ])
    background_pixels = np.zeros_like(cluster_image, dtype=bool)
    for i in range(image_height):
        for j in range(image_width):
            neighbor_coords = neighborhood_window + np.array([i, j])
            neighbor_coords = neighbor_coords[
                (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < image_height) &
                (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < image_width)
            ]
            neighbor_values = cluster_image[neighbor_coords[:, 0], neighbor_coords[:, 1]]
            if np.sum(neighbor_values) <= neighbor_threshold:
                background_pixels[i, j] = True
    return background_pixels

def interpolate_background(image, background_pixels):
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    interp_func = interp2d(x, y, image * background_pixels, kind='linear')
    background_image = interp_func(x, y)
    return background_image

def analyze_signal_background(
    signal_center, signal_width, signal_height,
    background_center, background_width, background_height,
    data, bin_boundaries, hist_start_bin, neighbor_threshold, cluster_threshold=0.2
):
    histograms = calculate_histograms(data, bin_boundaries, hist_start_bin)
    average_histogram = get_average_roi_histogram(
        histograms,
        background_center[0] - background_width // 2, background_center[0] + background_width // 2,
        background_center[1] - background_height // 2, background_center[1] + background_height // 2
    )
    emd_values = calculate_emd_values(histograms, average_histogram)
    cluster_image = (emd_values > cluster_threshold)
    background_pixels = identify_background_pixels(cluster_image, neighbor_threshold)
    integrated_image = np.sum(data, axis=0)
    background_image = interpolate_background(integrated_image, background_pixels)
    background_subtracted_image = integrated_image - background_image
    return {
        'background_pixels': background_pixels,
        'integrated_image': integrated_image,
        'background_image': background_image,
        'background_subtracted_image': background_subtracted_image
    }

def visualize_analysis(analysis_results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(analysis_results['background_pixels'], cmap='gray', origin='lower')
    axes[0].set_title('Background Pixels')
    axes[1].imshow(analysis_results['integrated_image'], cmap='gray', origin='lower')
    axes[1].set_title('Integrated Image')
    axes[2].imshow(analysis_results['background_image'], cmap='gray', origin='lower')
    axes[2].set_title('Background Image')
    axes[3].imshow(analysis_results['background_subtracted_image'], cmap='gray', origin='lower')
    axes[3].set_title('Background Subtracted Image')
    plt.show()

