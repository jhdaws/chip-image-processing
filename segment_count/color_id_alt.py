import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# --- Settings ---
IMG_DIR = "./image_in"
MASK_DIR = "./image_out"
OUT_DIR = "./analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Standard BGR color references
REF_BGR = {
    "red":   (0, 0, 255),
    "blue":  (255, 0, 0),
    "white": (255, 255, 255),
    "green": (0, 255, 0),
}

def match_color(avg_bgr):
    min_dist = float("inf")
    closest = "unknown"
    for color, ref_bgr in REF_BGR.items():
        dist = np.linalg.norm(np.array(avg_bgr) - np.array(ref_bgr))
        if dist < min_dist:
            min_dist = dist
            closest = color
    return closest

def get_color_from_selection(roi):
    selection = {}

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x0, x1 = sorted([x1, x2])
        y0, y1 = sorted([y1, y2])
        region = roi[y0:y1+1, x0:x1+1]
        avg_color = np.mean(region.reshape(-1, 3), axis=0)
        selection["color"] = tuple(np.round(avg_color, 1))
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title("Drag to select region for color (release to capture)")
    rect_selector = RectangleSelector(
        ax, onselect,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=False  # no handles, just drag+release
    )
    plt.show()

    return selection.get("color", None)

def main():
    print("=== STACK COLOR PICKER (STANDARD BGR MATCHING) ===")
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

    annotated = img.copy()

    print("\n=== STACK COLORS ===")
    for i, cnt in enumerate(contours):
        temp_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
        temp_mask = temp_mask.astype(bool)

        ys, xs = np.where(temp_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        roi = img[y0:y1+1, x0:x1+1]

        print(f"\n🟦 Stack #{i+1} — Select color region:")
        avg_bgr = get_color_from_selection(roi)

        if avg_bgr is not None:
            color_name = match_color(avg_bgr)
            print(f"Stack #{i+1} @ ({x0}, {y0}): BGR = {avg_bgr} → {color_name}")
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(annotated, color_name, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            print(f"Stack #{i+1} skipped.")

    out_path = os.path.join(OUT_DIR, f"classified_standard_colors_{image_name}")
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved result to {out_path}")

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Selected + Standard Color Classification")
    plt.show()

if __name__ == "__main__":
    main()
