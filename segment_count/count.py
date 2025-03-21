import os
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.cluster import KMeans

# Define paths
original_image_folder = "./image_in"  # Folder containing original images
segmented_image_folder = "./image_out"  # Folder containing segmented images

# Function to analyze a segmented image and extract colors from the original image
def analyze_chips(original_image_path, segmented_image_path):
    # Load the original image and segmented mask
    original_image = Image.open(original_image_path).convert("RGB")
    segmented_image = Image.open(segmented_image_path).convert("RGB")

    # Convert images to numpy arrays
    original_np = np.array(original_image)
    segmented_np = np.array(segmented_image)

    # Get unique colors in the segmented mask (excluding black background)
    unique_colors = np.unique(segmented_np.reshape(-1, 3), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]  # Remove black background

    # Analyze each chip
    chip_colors = []
    for color in unique_colors:
        # Create a mask for the current chip
        mask = np.all(segmented_np == color, axis=-1)

        # Extract the region from the original image using the mask
        chip_region = original_np[mask]

        # Use KMeans to find the dominant color in the chip region
        if len(chip_region) > 0:  # Ensure the region is not empty
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(chip_region)
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            chip_colors.append(dominant_color)

    # Print results
    print(f"Analyzing image: {original_image_path}")
    print(f"Number of chips detected: {len(chip_colors)}")
    for i, color in enumerate(chip_colors):
        print(f"Chip {i + 1}: Dominant Color = {color}")

    return chip_colors

# Process each image pair (original and segmented)
for image_name in os.listdir(original_image_folder):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue  # Skip non-image files

    # Define paths
    original_image_path = os.path.join(original_image_folder, image_name)
    segmented_image_path = os.path.join(segmented_image_folder, f"segmented_{image_name}")

    # Analyze the chips
    if os.path.exists(segmented_image_path):
        analyze_chips(original_image_path, segmented_image_path)
    else:
        print(f"Segmented image not found for: {image_name}")