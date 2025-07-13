import cv2
import os

# Define the folder paths
input_folder = '/home/UNT/sd1260/OSTrack_new4A6000_2/assets/ICRA_frames'  # Folder with your original images
left_output_folder = '/home/UNT/sd1260/OSTrack_new4A6000_2/assets/ours_icra'  # Folder for left parts
right_output_folder = '/home/UNT/sd1260/OSTrack_new4A6000_2/assets/ostrack_icra'  # Folder for right parts

# Create output folders if they do not exist
os.makedirs(left_output_folder, exist_ok=True)
os.makedirs(right_output_folder, exist_ok=True)

# Iterate over the frame range
for frame_index in range(420, 1340):
    frame_name = f'frame_{frame_index:04d}.png'
    input_image_path = os.path.join(input_folder, frame_name)
    
    # Read the image
    image = cv2.imread(input_image_path)
    
    if image is None:
        continue
    
    # Split the image into left and right parts
    left_part = image[:, :1280]  # Left part: first 1280 pixels
    right_part = image[:, 1280:]  # Right part: remaining 1280 pixels
    
    # Save the left and right parts to their respective folders
    cv2.imwrite(os.path.join(left_output_folder, f'left_{frame_name}'), left_part)
    cv2.imwrite(os.path.join(right_output_folder, f'right_{frame_name}'), right_part)

    print(f'Processed: {frame_name}')

print('Processing complete.')
