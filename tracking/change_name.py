import os
import glob
import shutil

# Define the source folder
source_folder = "/home/UNT/sd1260/OSTrack_new4A6000_2/assets/yoyo-19-image"  # Change to your actual source folder path
# source_folder = "/home/UNT/sd1260/OSTrack_new4A6000_2/assets/yoyo-19-att"  # Change to your actual source folder path

# Define the destination folder (appends "_new" to the source folder name)
destination_folder = f"{source_folder}_new"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get a sorted list of all PNG images
image_files = sorted(glob.glob(os.path.join(source_folder, "image_*.png")))
# image_files = sorted(glob.glob(os.path.join(source_folder, "pow_mean_att3_*.png")))

# Rename and move the images sequentially
for idx, image_path in enumerate(image_files, start=1):
    new_name = f"image_{idx}.png"
    # new_name = f"pow_mean_att3_{idx}.png"
    new_path = os.path.join(destination_folder, new_name)
    shutil.move(image_path, new_path)

print(f"Renaming and moving completed! Images are now in {destination_folder}")
