import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Directories
IMG_DIR = "./image_in"
MASK_DIR = "./image_out"
OUT_DIR = "./analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

def analyze_stack(img, mask):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    roi = img[y0:y1+1, x0:x1+1]
    mask_roi = mask[y0:y1+1, x0:x1+1]

    # Compute average BGR over masked region
    masked_pixels = roi[mask_roi]
    avg_bgr = np.mean(masked_pixels, axis=0)
    return tuple(np.round(avg_bgr, 1)), (x0, y0, x1, y1)

def main():
    print("=== STACK COLOR IDENTIFIER ===")
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

    for cnt in contours:
        temp_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
        temp_mask = temp_mask.astype(bool)
        avg_bgr, (x0, y0, x1, y1) = analyze_stack(img, temp_mask)
        summary[(x0, y0)] = avg_bgr
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(annotated, f"{tuple(map(int, avg_bgr))}", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    print("\n=== STACK COLORS ===")
    for pos, bgr in summary.items():
        print(f"Stack @ {pos}: BGR = {bgr}")

    out_path = os.path.join(OUT_DIR, f"color_tagged_{image_name}")
    cv2.imwrite(out_path, annotated)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Identified Stack Colors")
    plt.show()

if __name__ == "__main__":
    main()
