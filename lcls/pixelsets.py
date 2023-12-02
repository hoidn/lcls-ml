import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.patches import Arrow

from scipy.ndimage import center_of_mass

def find_center_of_mass(infilled_clusters):
    """
    Calculate the center of mass of the white signal clump in a given 2D array (infilled_clusters).
    
    Parameters:
    infilled_clusters (ndarray): 2D array representing the white signal clump and background.
    
    Returns:
    tuple: The center of mass coordinates (y, x).
    """
    return center_of_mass(infilled_clusters)


def correct_angle_assignment(i, j, center_of_mass, num_spokes=40):
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    angle = np.arctan2(i - center_of_mass[0], j - center_of_mass[1]) % (2 * np.pi)
    nearest_spoke_angle = min(angles, key=lambda x: abs(x - angle))
    return nearest_spoke_angle


def visualize_corrected_scale_values_heatmap(center_of_mass, infilled_clusters, num_spokes=20):
    """
    Visualize the corrected scale values for each spoke in j, k space using a heatmap.
    
    Parameters:
    center_of_mass (tuple): The (y, x) coordinates of the center of mass.
    infilled_clusters (ndarray): 2D array representing the signal clump and background.
    num_spokes (int): The number of spokes to visualize.
    """
    rows, cols = infilled_clusters.shape
    corrected_scale_heatmap = np.zeros((rows, cols))
    
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    
    for i in range(rows):
        for j in range(cols):
            angle = np.arctan2(i - center_of_mass[0], j - center_of_mass[1]) % (2 * np.pi)
            nearest_spoke_angle = min(angles, key=lambda x: min(abs(x - angle), 2 * np.pi - abs(x - angle)))
            scale = calculate_spoke_scale(center_of_mass, infilled_clusters, nearest_spoke_angle)
            if scale:
                corrected_scale_heatmap[i, j] = scale
                
    plt.imshow(corrected_scale_heatmap, cmap='viridis')
    plt.title('Heatmap of Corrected Scale Values in j, k Space')
    plt.xlabel('j')
    plt.ylabel('k')
    plt.colorbar(label='Scale Value')
    plt.show()





def revised_2x2_visualize_analysis_stages(center_of_mass, infilled_clusters, custom_distances, scales, num_spokes=20):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    axs[0, 0].imshow(infilled_clusters, cmap='gray')
    axs[0, 0].set_title('Infilled Clusters')
    axs[0, 0].set_xlabel('j')
    axs[0, 0].set_ylabel('k')
    
    axs[0, 1].imshow(custom_distances, cmap='viridis')
    axs[0, 1].set_title('Custom Radial Distances')
    axs[0, 1].set_xlabel('j')
    axs[0, 1].set_ylabel('k')
    
    axs[1, 0].imshow(infilled_clusters, cmap='gray')
    axs[1, 0].set_title('Spokes for Scale Calculation')
    axs[1, 0].set_xlabel('j')
    axs[1, 0].set_ylabel('k')
    
    axs[1, 1].imshow(np.tile(scales, (10, 1)), cmap='viridis', aspect='auto')
    axs[1, 1].set_title('Heatmap of Scale Values')
    axs[1, 1].set_xlabel('Spoke Index')
    axs[1, 1].set_ylabel('Replica')
    
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    
    for angle in angles:
        scale = calculate_spoke_scale(center_of_mass, infilled_clusters, angle)
        if scale:
            end_y, end_x = int(center_of_mass[0] + scale * np.sin(angle)), int(center_of_mass[1] + scale * np.cos(angle))
            axs[1, 0].add_patch(Arrow(center_of_mass[1], center_of_mass[0], end_x - center_of_mass[1], end_y - center_of_mass[0], width=0.5, color='red'))
    
    plt.tight_layout()
    plt.show()


def calculate_pixel_sets_updated(custom_distances, N):
    rows, cols = custom_distances.shape
    pixel_sets = []
    sorted_indices = np.dstack(np.unravel_index(np.argsort(custom_distances.ravel()), (rows, cols)))[0]
    start_idx = 0
    while start_idx < sorted_indices.shape[0]:
        end_idx = start_idx + N
        pixel_set_indices = sorted_indices[start_idx:end_idx]
        pixel_set = np.zeros((rows, cols), dtype = int)
        for idx in pixel_set_indices:
            pixel_set[idx[0], idx[1]] = 1
        pixel_sets.append(pixel_set)
        start_idx = end_idx
    return np.array(pixel_sets)


def run_radial_analysis(infilled_clusters, num_spokes=40, num_pixels_per_set=100):
    """
    Run the radial analysis on the given 2D binary mask (infilled_clusters).
    
    Parameters:
        infilled_clusters (numpy.ndarray): 2D binary mask representing the infilled clusters.
        num_spokes (int): Number of spokes to use for radial analysis.
        num_pixels_per_set (int): Number of pixels per pixel set.
    
    Returns:
        tuple: (custom_distances, pixel_sets, pixel_set_indices, scale_values)
    """
    center_of_mass = ndimage.measurements.center_of_mass(infilled_clusters)
    
    angles = np.linspace(0, 2*np.pi, num_spokes, endpoint=False)
    scales = [calculate_spoke_scale(center_of_mass, infilled_clusters, angle) for angle in angles]
    
    custom_distances = calculate_custom_distances(center_of_mass, infilled_clusters, num_spokes=num_spokes)
    
    pixel_sets = calculate_pixel_sets_updated(custom_distances, num_pixels_per_set)
    
    pixel_set_indices = np.zeros_like(infilled_clusters, dtype=int)
    
    for i, pixel_set in enumerate(pixel_sets):
        pixel_set_indices += i * pixel_set
        
    scale_values = np.zeros_like(infilled_clusters, dtype=float)
    for i in range(infilled_clusters.shape[0]):
        for j in range(infilled_clusters.shape[1]):
            nearest_spoke_angle = correct_angle_assignment(i, j, center_of_mass, num_spokes=num_spokes)
            scale_values[i, j] = scales[int(nearest_spoke_angle // (2 * np.pi / num_spokes))]
            
    return custom_distances, pixel_sets, pixel_set_indices, scale_values

def calculate_spoke_scale(center_of_mass, infilled_clusters, angle):
    """
    Calculate the scale value along a spoke emanating from the center of mass at a given angle.
    The scale is defined as the distance from the center of mass to the first transition from 'signal' to 'background'.
    """
    i, j = int(center_of_mass[0]), int(center_of_mass[1])
    di, dj = np.sin(angle), np.cos(angle)
    scale = None
    while 0 <= i < infilled_clusters.shape[0] and 0 <= j < infilled_clusters.shape[1]:
        if infilled_clusters[int(i), int(j)] == 0:
            scale = np.sqrt((i - center_of_mass[0]) ** 2 + (j - center_of_mass[1]) ** 2)
            break
        i, j = i + di, j + dj
    return scale

def calculate_custom_distances(center_of_mass, infilled_clusters, num_spokes):
    """
    Calculate the custom radial distances based on the center of mass and infilled_clusters using the corrected angle assignment.
    """
    distances = np.zeros_like(infilled_clusters, dtype=float)
    angles = np.linspace(-np.pi, np.pi, num_spokes, endpoint=False)
    for i in range(infilled_clusters.shape[0]):
        for j in range(infilled_clusters.shape[1]):
            angle = np.arctan2(i - center_of_mass[0], j - center_of_mass[1])
            nearest_spoke_angle = min(angles, key=lambda x: abs(x - angle))
            scale = calculate_spoke_scale(center_of_mass, infilled_clusters, nearest_spoke_angle)
            if scale:
                distances[i, j] = np.sqrt((i - center_of_mass[0]) ** 2 + (j - center_of_mass[1]) ** 2) / scale
    return distances


def plot_radial_analysis(infilled_clusters, custom_distances, pixel_set_indices, scale_values, num_spokes=40):
    """
    Plot the results of the radial analysis as a 2x2 visualization.
    
    Parameters:
        infilled_clusters (numpy.ndarray): 2D binary mask representing the infilled clusters.
        custom_distances (numpy.ndarray): 2D array of custom radial distances.
        pixel_set_indices (numpy.ndarray): 2D array indicating the pixel set index for each pixel.
        scale_values (numpy.ndarray): 2D array indicating the scale value for each pixel.
        num_spokes (int): Number of spokes to use for radial analysis.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    axs[0, 0].imshow(pixel_set_indices * (pixel_set_indices < 20), cmap='tab20', interpolation='none')
    axs[0, 0].set_title('Pixel Sets by Index')
    axs[0, 0].set_xlabel('j')
    axs[0, 0].set_ylabel('k')
    
    axs[0, 1].imshow(custom_distances, cmap='hot', interpolation='none')
    axs[0, 1].set_title('Custom Radial Distances')
    axs[0, 1].set_xlabel('j')
    axs[0, 1].set_ylabel('k')
    
    axs[1, 0].imshow(scale_values, cmap='hot', interpolation='none')
    axs[1, 0].set_title('Scale Values')
    axs[1, 0].set_xlabel('j')
    axs[1, 0].set_ylabel('k')
    
    axs[1, 1].imshow(infilled_clusters, cmap='gray', interpolation='none')
    center_of_mass = ndimage.measurements.center_of_mass(infilled_clusters)
    angles = np.linspace(0, 2*np.pi, num_spokes, endpoint=False)
    for angle in angles:
        scale = calculate_spoke_scale(center_of_mass, infilled_clusters, angle)
        if scale:
            end_y, end_x = int(center_of_mass[0] + scale * np.sin(angle)), int(center_of_mass[1] + scale * np.cos(angle))
            axs[1, 1].add_patch(Arrow(center_of_mass[1], center_of_mass[0], end_x - center_of_mass[1], end_y - center_of_mass[0], width=0.5, color='red'))
    axs[1, 1].set_title('Spokes for Scale Calculation')
    axs[1, 1].set_xlabel('j')
    axs[1, 1].set_ylabel('k')
    
    plt.tight_layout()
    plt.show()

