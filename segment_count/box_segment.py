import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
from matplotlib.widgets import RectangleSelector
import os


# --- Setup ---
CHECKPOINT_PATH = "./model/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cpu"  # or "cuda" if you have GPU
input_folder = "./image_in"
output_folder = "./image_out"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load models
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

def process_image(image_name):
    # Full image path
    image_path = os.path.join(input_folder, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))

    # --- Interactive Box Selection ---
    def onselect(eclick, erelease):
        global box
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.title("Click and drag to draw a box around the poker chips")
    rect = plt.Rectangle((0, 0), 1, 1, fill=False, linewidth=2)
    ax.add_patch(rect)

    rs = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=5, minspany=5,  # Minimum box size
        spancoords='pixels',
        interactive=True
    )
    plt.show()

    # --- Run SAM with the box prompt ---
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],  # Add batch dimension
        multimask_output=False,
    )

    # --- Show results ---
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Original")
    plt.subplot(122)
    plt.imshow(masks[0], cmap='gray')
    plt.title("Segmentation Mask")
    plt.show()

    # Save mask
    binary_mask = (masks[0] * 255).astype(np.uint8)  # Convert to 8-bit grayscale
    output_path = os.path.join(output_folder, f"segmented_{image_name}")
    mask_image = Image.fromarray(binary_mask, mode='L')  # 'L' for grayscale
    mask_image.save(output_path)
    print(f"Mask saved to: {output_path}")

# --- Main Loop ---
while True:
    image_name = input("Enter image name (include file type) or type 'exit' to quit: ")
    if image_name.lower() == 'exit':
        print("Exiting...")
        break
    process_image(image_name)