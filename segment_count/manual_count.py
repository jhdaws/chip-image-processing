import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Directories
IMG_DIR = "./image_in"
MASK_DIR = "./image_out"
OUT_DIR = "./analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

REF_BGR = {
    "red":   (45.0, 55.0, 100.0),
    "blue":  (60.0, 57.0, 64.0),
    "white": (87.0, 105.0, 119.0),
    "green": (67.0, 79.0, 81.0),
}

def match_color(avg_bgr):
    min_dist = float('inf')
    closest_color = "unknown"
    for color, ref_bgr in REF_BGR.items():
        dist = np.linalg.norm(np.array(avg_bgr) - np.array(ref_bgr))
        if dist < min_dist:
            min_dist = dist
            closest_color = color
    return closest_color

def get_user_measurements(img, title):
    coords = []

    def onselect(eclick, erelease):
        coords.append((eclick.ydata, erelease.ydata, eclick.xdata, erelease.xdata))
        plt.close()

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title + "\nClick and drag vertically or horizontally")
    from matplotlib.widgets import RectangleSelector
    rs = RectangleSelector(plt.gca(), onselect, useblit=True, button=[1], interactive=True)
    plt.show()

    if not coords:
        return None
    y1, y2, x1, x2 = coords[0]
    height = abs(y2 - y1)
    width = abs(x2 - x1)
    return height if height > width else width

def analyze_stack(img, mask):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    roi = img[y0:y1+1, x0:x1+1]
    mask_roi = mask[y0:y1+1, x0:x1+1]

    masked_pixels = roi[mask_roi]
    avg_bgr = np.mean(masked_pixels, axis=0)
    color = match_color(avg_bgr)

    print(f"\n🟩 STACK COLOR: {color}")
    chip_diameter = get_user_measurements(roi, "Measure TOP CHIP diameter")
    stack_height = get_user_measurements(roi, "Measure STACK height")
    chip_height = get_user_measurements(roi, "Measure ONE CHIP height")

    if None in (chip_diameter, stack_height, chip_height):
        print("Measurement cancelled. Skipping stack.")
        return 0, color, (x0, y0, x1, y1)

    count = int(round(stack_height / chip_height))
    return count, color, (x0, y0, x1, y1)

def main():
    print("=== CHIP COUNTER (Manual-Assisted Mode) ===")
    image_name = input("Image filename (in ./image_in): ").strip()
    img_path = os.path.join(IMG_DIR, image_name)
    mask_path = os.path.join(MASK_DIR, f"segmented_{image_name}")

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print("Missing image or segmentation mask. Ensure both files exist.")
        return

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    summary = {}
    annotated = img.copy()

    for i, cnt in enumerate(contours):
        print(f"\n🟦 Measuring stack #{i + 1}")
        temp_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
        temp_mask = temp_mask.astype(bool)
        count, color, (x0, y0, x1, y1) = analyze_stack(img, temp_mask)
        summary[color] = summary.get(color, 0) + count
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(annotated, f"{color}:{count}", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    print("\n=== FINAL CHIP COUNTS ===")
    for color, count in summary.items():
        print(f"{color:>6}: {count} chips")

    out_path = os.path.join(OUT_DIR, f"manual_count_{image_name}")
    cv2.imwrite(out_path, annotated)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Manual-Assisted Chip Counting")
    plt.show()

if __name__ == "__main__":
    main()
