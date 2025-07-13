import os
import cv2
import numpy as np

# Paths
base_data_path = "/home/UNT/sd1260/OSTrack/data/lasot/airplane/airplane-1"
save_path = "/home/UNT/sd1260/OSTrack_new4A6000_2/assets/tracking_result_final/airplane-1_croped5"

# Create save directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Search region parameters
SEARCH_FACTOR = 5.0
SEARCH_SIZE = 256  # Adjust this if needed

# Load ground truth bounding boxes
gt_file = os.path.join(base_data_path, "groundtruth.txt")
with open(gt_file, "r") as f:
    gt_bboxes = [list(map(int, line.strip().split(','))) for line in f]

# Get image file paths
img_folder = os.path.join(base_data_path, "img")
img_files = sorted(os.listdir(img_folder))

for i, (x, y, w, h) in enumerate(gt_bboxes):
    # Compute the center of the bounding box
    cx, cy = x + w // 2, y + h // 2

    # Compute search region size
    search_size = int(np.sqrt(w * h) + SEARCH_FACTOR * (w + h) / 2)

    # Load the image
    img_path = os.path.join(img_folder, img_files[i])
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Could not load {img_path}")
        continue

    H, W, _ = img.shape  # Image dimensions

    # Adjust the crop size if it exceeds the image boundaries
    if search_size > H or search_size > W:
        search_size = min(H, W)

    # Compute the crop coordinates
    x1, y1 = cx - search_size // 2, cy - search_size // 2
    x2, y2 = x1 + search_size, y1 + search_size

    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    # Crop the image
    cropped = img[y1:y2, x1:x2]

    # If the crop size is smaller than the desired size, pad to SEARCH_SIZE
    if cropped.shape[0] < SEARCH_SIZE or cropped.shape[1] < SEARCH_SIZE:
        top_padding = max(0, (SEARCH_SIZE - cropped.shape[0]) // 2)
        bottom_padding = max(0, SEARCH_SIZE - cropped.shape[0] - top_padding)
        left_padding = max(0, (SEARCH_SIZE - cropped.shape[1]) // 2)
        right_padding = max(0, SEARCH_SIZE - cropped.shape[1] - left_padding)

        # Pad the image to match the SEARCH_SIZE
        cropped = cv2.copyMakeBorder(cropped, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Resize to SEARCH_SIZE if needed
    cropped_resized = cv2.resize(cropped, (SEARCH_SIZE, SEARCH_SIZE))

    # Save the cropped image
    save_filename = os.path.join(save_path, f"crop_{i+1:04d}.jpg")
    cv2.imwrite(save_filename, cropped_resized)

    print(f"Saved: {save_filename}")

print("Cropping complete! 🚀")
