import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

# --- Configuration ---
CHECKPOINT_PATH = "./model/sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
DEVICE = "cpu"  # or "cuda" if you have GPU
INPUT_FOLDER = "./image_in"
SEGMENTED_FOLDER = "./image_out"
OUTPUT_FOLDER = "./analysis_results"

# Color reference values
REF_BGR = {
    "red":   (45.0, 55.0, 100.0),
    "blue":  (60.0, 57.0, 64.0),
    "white": (87.0, 105.0, 119.0),
    "green": (67.0, 79.0, 81.0),
}

AUTO_TOP_CHIP_RATIO = 0.7  # used for automatic correction

# Ensure output folders exist
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load SAM model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# --- Segmentation Functions ---
def segment_image(image_name):
    # Full image path
    image_path = os.path.join(INPUT_FOLDER, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

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

    # Save mask
    binary_mask = (masks[0] * 255).astype(np.uint8)  # Convert to 8-bit grayscale
    output_path = os.path.join(SEGMENTED_FOLDER, f"segmented_{image_name}")
    mask_image = Image.fromarray(binary_mask, mode='L')  # 'L' for grayscale
    mask_image.save(output_path)
    print(f"Mask saved to: {output_path}")
    
    return binary_mask, image

# --- Counting Functions ---
def match_color(avg_bgr):
    min_dist = float('inf')
    closest_color = "unknown"
    for color, ref_bgr in REF_BGR.items():
        dist = np.linalg.norm(np.array(avg_bgr) - np.array(ref_bgr))
        if dist < min_dist:
            min_dist = dist
            closest_color = color
    return closest_color

def get_measurement_from_drag(img, prompt):
    coords = []

    def onselect(eclick, erelease):
        height = abs(eclick.ydata - erelease.ydata)
        coords.append(height)
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(prompt)
    rect = RectangleSelector(ax, onselect, useblit=True, button=[1],
                             interactive=True, minspanx=5, minspany=5)
    plt.show()

    return coords[0] if coords else None

def get_chip_height_from_stack(img, mask):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    roi = img[y0:y1+1, x0:x1+1]

    return get_measurement_from_drag(roi, "Drag from TOP to BOTTOM of ONE chip (Zoomed View)")

def get_top_chip_visibility(roi):
    return get_measurement_from_drag(roi, "Drag from TOP to BOTTOM of visible top chip surface")

def analyze_stack(img, mask, chip_height, use_manual_top_correction):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    roi = img[y0:y1+1, x0:x1+1]
    mask_roi = mask[y0:y1+1, x0:x1+1]

    masked_pixels = roi[mask_roi]
    avg_bgr = np.mean(masked_pixels, axis=0)
    color = match_color(avg_bgr)

    stack_height = y1 - y0

    if use_manual_top_correction:
        top_visible = get_top_chip_visibility(roi)
        if top_visible is None:
            return 0, color, (x0, y0, x1, y1)
        adjusted_height = stack_height - top_visible
    else:
        adjusted_height = stack_height - AUTO_TOP_CHIP_RATIO * chip_height

    count = int(round(adjusted_height / chip_height))
    return max(1, count), color, (x0, y0, x1, y1)

def count_chips(image_name, mask, original_image):
    img = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_cnt = max(contours, key=cv2.contourArea)
    zoom_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(zoom_mask, [largest_cnt], -1, 255, -1)
    zoom_mask = zoom_mask.astype(bool)

    chip_height = get_chip_height_from_stack(img, zoom_mask)
    if chip_height is None:
        print("Chip height calibration failed.")
        return

    print(f"✔️ Chip height calibrated at {chip_height:.1f} px")

    use_manual_top_correction = input("\nWould you like to manually correct for the visible top chip? (y/n): ").lower().strip() == 'y'

    summary = {}
    annotated = img.copy()

    for i, cnt in enumerate(contours):
        print(f"\n🟦 Measuring stack #{i + 1}")
        temp_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
        temp_mask = temp_mask.astype(bool)
        count, color, (x0, y0, x1, y1) = analyze_stack(img, temp_mask, chip_height, use_manual_top_correction)
        summary[color] = summary.get(color, 0) + count
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(annotated, f"{color}:{count}", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    print("\n=== FINAL CHIP COUNTS ===")
    for color, count in summary.items():
        print(f"{color:>6}: {count} chips")

    out_path = os.path.join(OUTPUT_FOLDER, f"interactive_count_{image_name}")
    cv2.imwrite(out_path, annotated)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Counted Stacks (Manual Toggle)")
    plt.show()

# --- Main Program ---
def main():
    print("=== POKER CHIP COUNTER ===")
    print("1. Segment the image")
    print("2. Count the chips")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            image_name = input("Enter image name (include file type): ")
            mask, original_image = segment_image(image_name)
            if mask is not None:
                print("Segmentation completed successfully!")
                
        elif choice == '2':
            if 'image_name' not in locals():
                image_name = input("Enter image name (include file type): ")
                mask_path = os.path.join(SEGMENTED_FOLDER, f"segmented_{image_name}")
                if not os.path.exists(mask_path):
                    print("No segmented image found. Please segment first.")
                    continue
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                original_image = np.array(Image.open(os.path.join(INPUT_FOLDER, image_name)).convert("RGB"))
            
            count_chips(image_name, mask, original_image)
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()