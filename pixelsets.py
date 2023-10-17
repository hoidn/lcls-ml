import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from matplotlib.patches import Arrow

# Importing required functions from scipy.ndimage
from scipy.ndimage import center_of_mass

# Define a function to calculate the center of mass of the white signal clump in infilled_clusters
def find_center_of_mass(infilled_clusters):
    """
    Calculate the center of mass of the white signal clump in a given 2D array (infilled_clusters).
    
    Parameters:
    infilled_clusters (ndarray): 2D array representing the white signal clump and background.
    
    Returns:
    tuple: The center of mass coordinates (y, x).
    """
    return center_of_mass(infilled_clusters)

# # Calculate the center of mass for the given infilled_clusters data
# center_of_mass_coordinates = find_center_of_mass(infilled_clusters)

# def assign_angle(i, j, center_of_mass, num_spokes=20):
#     """
#     Correct the angle assignment to fall within [0, 2*pi] range and find the nearest spoke angle.
#     """
#     angle = np.arctan2(i - center_of_mass[0], j - center_of_mass[1]) % (2 * np.pi)
#     angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
#     nearest_spoke_angle = min(angles, key=lambda x: min(abs(x - angle), 2 * np.pi - abs(x - angle)))
#     return nearest_spoke_angle

# Function to correct the angle assignment
def correct_angle_assignment(i, j, center_of_mass, num_spokes=40):
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    angle = np.arctan2(i - center_of_mass[0], j - center_of_mass[1]) % (2 * np.pi)
    nearest_spoke_angle = min(angles, key=lambda x: abs(x - angle))
    return nearest_spoke_angle


# def calculate_spoke_scale(center_of_mass, infilled_clusters, angle):
#     """
#     Calculate the scale for a given spoke angle based on the center of mass and infilled_clusters.
#     """
#     y, x = center_of_mass
#     rows, cols = infilled_clusters.shape
#     for scale in range(1, max(rows, cols)):
#         end_y, end_x = int(y + scale * np.sin(angle)), int(x + scale * np.cos(angle))
#         if 0 <= end_y < rows and 0 <= end_x < cols:
#             if infilled_clusters[end_y, end_x] == 0:
#                 return scale
#     return None

# def calculate_spoke_scale(center_of_mass, infilled_clusters, angle):
#     """
#     Calculate the scale value along a spoke emanating from the center of mass at a given angle.
#     The scale is defined as the distance from the center of mass to the first transition from 'signal' to 'background'.
#     """
#     i, j = int(center_of_mass[0]), int(center_of_mass[1])
#     di, dj = np.sin(angle), np.cos(angle)
#     scale = None
#     while 0 <= i < infilled_clusters.shape[0] and 0 <= j < infilled_clusters.shape[1]:
#         if infilled_clusters[int(i), int(j)] == 0:
#             scale = np.sqrt((i - center_of_mass[0]) ** 2 + (j - center_of_mass[1]) ** 2)
#             break
#         i, j = i + di, j + dj
#     return scale

# def calculate_custom_distances(center_of_mass, infilled_clusters, num_spokes=20):
#     """
#     Calculate the custom radial distances based on the center of mass and infilled_clusters using the corrected angle assignment.
#     """
#     rows, cols = infilled_clusters.shape
#     custom_distances = np.zeros((rows, cols))
#     for i in range(rows):
#         for j in range(cols):
#             nearest_spoke_angle = correct_angle_assignment(i, j, center_of_mass, num_spokes)
#             scale = calculate_spoke_scale(center_of_mass, infilled_clusters, nearest_spoke_angle)
#             if scale:
#                 euclidean_distance = np.sqrt((i - center_of_mass[0])**2 + (j - center_of_mass[1])**2)
#                 custom_distances[i, j] = euclidean_distance / scale
#     return custom_distances

# Function to visualize the corrected scale values in j, k space using a heatmap
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
    
    # Define angles for the spokes
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    
    for i in range(rows):
        for j in range(cols):
            # Correct angle assignment to [0, 2*pi] range
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

## Visualize the corrected scale values in j, k space using a heatmap for the new uploaded infilled_clusters
#visualize_corrected_scale_values_heatmap(center_of_mass_coordinates, infilled_clusters, num_spokes=20)


# # Recalculate the center of mass for the new uploaded infilled_clusters
# center_of_mass_coordinates = center_of_mass(infilled_clusters)

# # Recalculate the custom radial distances and scales using the new uploaded infilled_clusters
# custom_distances_new_uploaded = calculate_custom_distances(center_of_mass_coordinates, infilled_clusters, num_spokes=20)
# scales_new_uploaded_20_spokes = [calculate_spoke_scale(center_of_mass_coordinates, infilled_clusters, angle) for angle in angles_20_spokes]

# Revised 2x2 visualization layout with new uploaded infilled_clusters and 20 spokes
def revised_2x2_visualize_analysis_stages(center_of_mass, infilled_clusters, custom_distances, scales, num_spokes=20):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Visualize infilled_clusters
    axs[0, 0].imshow(infilled_clusters, cmap='gray')
    axs[0, 0].set_title('Infilled Clusters')
    axs[0, 0].set_xlabel('j')
    axs[0, 0].set_ylabel('k')
    
    # Visualize custom_distances
    axs[0, 1].imshow(custom_distances, cmap='viridis')
    axs[0, 1].set_title('Custom Radial Distances')
    axs[0, 1].set_xlabel('j')
    axs[0, 1].set_ylabel('k')
    
    # Visualize spokes and infilled_clusters
    axs[1, 0].imshow(infilled_clusters, cmap='gray')
    axs[1, 0].set_title('Spokes for Scale Calculation')
    axs[1, 0].set_xlabel('j')
    axs[1, 0].set_ylabel('k')
    
    # Heatmap of scale values
    axs[1, 1].imshow(np.tile(scales, (10, 1)), cmap='viridis', aspect='auto')
    axs[1, 1].set_title('Heatmap of Scale Values')
    axs[1, 1].set_xlabel('Spoke Index')
    axs[1, 1].set_ylabel('Replica')
    
    angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    
    # Draw the spokes
    for angle in angles:
        scale = calculate_spoke_scale(center_of_mass, infilled_clusters, angle)
        if scale:
            end_y, end_x = int(center_of_mass[0] + scale * np.sin(angle)), int(center_of_mass[1] + scale * np.cos(angle))
            axs[1, 0].add_patch(Arrow(center_of_mass[1], center_of_mass[0], end_x - center_of_mass[1], end_y - center_of_mass[0], width=0.5, color='red'))
    
    plt.tight_layout()
    plt.show()

## Visualize the revised analysis stages with new uploaded infilled_clusters and 20 spokes in 2x2 layout
#revised_2x2_visualize_analysis_stages(center_of_mass_coordinates, infilled_clusters, custom_distances_new_uploaded, scales_new_uploaded_20_spokes, num_spokes=20)

# Function to calculate pixel sets
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


# High-level functions for analysis and visualization
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
    # Calculate the center of mass
    center_of_mass = ndimage.measurements.center_of_mass(infilled_clusters)
    
    # Calculate scales using the original version of calculate_spoke_scale
    angles = np.linspace(0, 2*np.pi, num_spokes, endpoint=False)
    scales = [calculate_spoke_scale(center_of_mass, infilled_clusters, angle) for angle in angles]
    
    # Calculate custom radial distances
    custom_distances = calculate_custom_distances(center_of_mass, infilled_clusters, num_spokes=num_spokes)
    
    # Calculate pixel sets
    pixel_sets = calculate_pixel_sets_updated(custom_distances, num_pixels_per_set)
    
    # Initialize an empty array to store the pixel set index for each pixel
    pixel_set_indices = np.zeros_like(infilled_clusters, dtype=int)
    
    # Assign the index of the corresponding pixel set to each pixel
    for i, pixel_set in enumerate(pixel_sets):
        pixel_set_indices += i * pixel_set
        
    # Generate the scale heatmap
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

# def run_radial_analysis(infilled_clusters, num_spokes=40, num_pixels_per_set=100):
#     center_of_mass = ndimage.measurements.center_of_mass(infilled_clusters)
#     custom_distances = calculate_custom_distances(center_of_mass, infilled_clusters, num_spokes)
#     pixel_sets = calculate_pixel_sets_updated(custom_distances, num_pixels_per_set)
#     pixel_set_indices = np.zeros_like(custom_distances, dtype=int)
#     for i, pixel_set in enumerate(pixel_sets):
#         pixel_set_indices[pixel_set] = i
#     scale_values = np.zeros_like(custom_distances)
#     angles = np.linspace(-np.pi, np.pi, num_spokes, endpoint=False)
#     for i in range(infilled_clusters.shape[0]):
#         for j in range(infilled_clusters.shape[1]):
#             angle = np.arctan2(i - center_of_mass[0], j - center_of_mass[1])
#             nearest_spoke_angle = min(angles, key=lambda x: abs(x - angle))
#             scale_values[i, j] = calculate_spoke_scale(center_of_mass, infilled_clusters, nearest_spoke_angle)
#     return custom_distances, pixel_sets, pixel_set_indices, scale_values

# # Debugging the run_radial_analysis function by printing out some internal variables
# def run_radial_analysis(infilled_clusters, num_spokes=20, num_pixels_per_set=50):
#     """
#     Run the entire radial analysis and return the custom distances, pixel sets, and scale values.
#     This is a debug version of the function to investigate its behavior.
#     """
#     # Calculate the center of mass
#     center_of_mass = np.array(np.round(np.mean(np.argwhere(infilled_clusters), axis=0)), dtype=int)

#     # Calculate custom distances
#     custom_distances = calculate_custom_distances(center_of_mass, infilled_clusters, num_spokes)

#     # Calculate the pixel sets
#     pixel_sets = calculate_pixel_sets_updated(custom_distances, num_pixels_per_set)

#     # Initialize an empty array to store the pixel set index for each pixel
#     pixel_set_indices = np.zeros_like(infilled_clusters, dtype=int)

#     # Assign the index of the corresponding pixel set to each pixel
#     for i, pixel_set in enumerate(pixel_sets):
#         # Convert pixel_set to a NumPy array for easier indexing
#         pixel_set = np.array(pixel_set)
#         pixel_set_indices[pixel_set[:, 0], pixel_set[:, 1]] = i

#     # Calculate the scale values
#     scale_values = np.zeros_like(infilled_clusters, dtype=float)
#     angles = np.linspace(-np.pi, np.pi, num_spokes, endpoint=False)
#     for i in range(infilled_clusters.shape[0]):
#         for j in range(infilled_clusters.shape[1]):
#             angle = np.arctan2(i - center_of_mass[0], j - center_of_mass[1])
#             nearest_spoke_angle = min(angles, key=lambda x: abs(x - angle))
#             scale = calculate_spoke_scale(center_of_mass, infilled_clusters, nearest_spoke_angle)
#             if scale:
#                 scale_values[i, j] = scale

#     return custom_distances, pixel_sets, pixel_set_indices, scale_values

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
    
    # Pixel sets by index heatmap
    axs[0, 0].imshow(pixel_set_indices * (pixel_set_indices < 20), cmap='tab20', interpolation='none')
    axs[0, 0].set_title('Pixel Sets by Index')
    axs[0, 0].set_xlabel('j')
    axs[0, 0].set_ylabel('k')
    
    # Custom radial distances heatmap
    axs[0, 1].imshow(custom_distances, cmap='hot', interpolation='none')
    axs[0, 1].set_title('Custom Radial Distances')
    axs[0, 1].set_xlabel('j')
    axs[0, 1].set_ylabel('k')
    
    # Scale heatmap
    axs[1, 0].imshow(scale_values, cmap='hot', interpolation='none')
    axs[1, 0].set_title('Scale Values')
    axs[1, 0].set_xlabel('j')
    axs[1, 0].set_ylabel('k')
    
    # Spokes for scale calculation figure
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

## Run the high-level functions
#custom_distances_original, pixel_sets_original, pixel_set_indices_original, scale_values_original = run_radial_analysis(infilled_clusters)
#plot_radial_analysis(infilled_clusters, custom_distances_original, pixel_set_indices_original, scale_values_original)
