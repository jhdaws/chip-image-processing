import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from PIL import Image

# Define paths
input_folder = "./image_in"
output_folder = "./image_out"
checkpoint_path = "./model/sam_vit_b_01ec64.pth"
model_type = "vit_b"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the SAM model
print("Loading SAM model...")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    min_mask_region_area=100,
)
print("Model loaded successfully.")

# Prompt for the image name
image_name = input("Enter the name of the image to process (with extension): ")

# Process the specified image
image_path = os.path.join(input_folder, image_name)
if not os.path.exists(image_path):
    print(f"Image {image_name} does not exist in the input folder.")
else:
    print(f"Processing image: {image_name}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))  # Uniform size for consistent processing
    image_np = np.array(image)

    # Generate masks
    print("Generating masks...")
    masks = mask_generator.generate(image_np)
    print(f"Generated {len(masks)} masks.")

    # Create binary mask (single channel)
    binary_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)  # Black background
    for mask in masks:
        binary_mask[mask["segmentation"]] = 255  # White for poker chips

    # Save as grayscale image
    output_image = Image.fromarray(binary_mask, mode='L')  # 'L' for 8-bit grayscale
    output_path = os.path.join(output_folder, f"segmented_{image_name}")
    output_image.save(output_path)
    print(f"Saved binary mask to: {output_path}")

print("Processing completed.")