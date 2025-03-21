import os
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from PIL import Image

# Define paths
input_folder = "./image_in"
output_folder = "./image_out"
checkpoint_path = "./model/sam_vit_b_01ec64.pth"  # Path to the vit_b checkpoint
model_type = "vit_b"  # Use the smallest and fastest model

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the SAM model
print("Loading SAM model...")
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,  # Fewer points for faster processing
    pred_iou_thresh=0.88,  # Filter low-quality masks
    stability_score_thresh=0.92,  # Filter unstable masks
    crop_n_layers=0,  # Disable crop layers
    min_mask_region_area=100,  # Filter small masks
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
    image = image.resize((512, 512))  # Resize to reduce memory usage
    image_np = np.array(image)

    # Generate masks for the entire image
    print("Generating masks...")
    masks = mask_generator.generate(image_np)
    print(f"Generated {len(masks)} masks.")

    # Create a color mask (RGB)
    color_mask = np.zeros_like(image_np)  # Initialize an empty RGB mask
    for mask in masks:
        # Assign a random color to each mask
        color = np.random.randint(0, 256, size=(3,))  # Random RGB color
        color_mask[mask["segmentation"]] = color  # Apply the color to the mask

    # Convert the color mask to an image
    color_mask_image = Image.fromarray(color_mask.astype(np.uint8))

    # Save the segmented image
    output_path = os.path.join(output_folder, f"segmented_{image_name}")
    color_mask_image.save(output_path)
    print(f"Saved segmented image to: {output_path}")

print("Image processing completed.")