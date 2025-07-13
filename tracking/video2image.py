import os
import cv2

video_path = "/home/UNT/sd1260/OSTrack_new4A6000_2/assets/ICRA25_video.mp4"  # Change to your video file
output_folder = "/home/UNT/sd1260/OSTrack_new4A6000_2/assets/ICRA_frames"  # Folder to save frames

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")  # Save frames sequentially
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames and saved in '{output_folder}'")
