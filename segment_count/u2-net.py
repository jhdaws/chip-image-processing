import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from u2net import U2NET  # Install with: pip install u2net

# Define paths
input_folder = "./image_in"
output_folder = "./image_out"
os.makedirs(output_folder, exist_ok=True)

# Load U^2-Net model
model = U2NET(3, 1)
model.load_state_dict(torch.load("u2net.pth", map_location="cpu"))
model.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process an image
image_name = input("Enter image name (e.g., chips.jpg): ")
image_path = os.path.join(input_folder, image_name)

if not os.path.exists(image_path):
    print("Image not found!")
else:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().cpu().numpy()

    # Convert to binary mask (0=background, 255=foreground)
    mask = (mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask).resize(image.size)

    # Apply mask to make background black
    result = image.copy()
    result.putalpha(mask)  # Transparent background
    result = result.convert("RGB")  # Optional: Force black background
    result.paste(0, mask=Image.fromarray(255 - np.array(mask)))  # Force black bg

    output_path = os.path.join(output_folder, f"segmented_{image_name}")
    result.save(output_path)
    print(f"Saved to {output_path}")