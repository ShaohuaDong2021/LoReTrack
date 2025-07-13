import cv2
import os

# Define the sequence name
# sequence_name = "airplane-1"
sequence_name = "racing-20"

# Define base paths
base_data_path = "/home/UNT/sd1260/OSTrack/data/lasot/racing"
base_output_loretrack = "/home/UNT/sd1260/OSTrack_new4A6000_2/output/test_70.3/tracking_results/ostrack/vitb_384_mae_32x4_ep300"
base_output_ostrack = "/home/UNT/sd1260/OSTrack_test_A6000/output/test_256baseline_68.53/tracking_results/ostrack/vitb_256_mae_32x4_ep300"
base_output_aiatrack = "/home/UNT/sd1260/OSTrack_new4A6000_2/tracker_rawresults/aiatrack/baseline"
base_output_mixformer = "/home/UNT/sd1260/OSTrack_new4A6000_2/tracker_rawresults/mixformer-22k-69.2/baseline_skip200_sear4.55"
base_output_romtrack = "/home/UNT/sd1260/OSTrack_new4A6000_2/tracker_rawresults/Romtracker-69.3/baseline"
base_output_simtrack = "/home/UNT/sd1260/OSTrack_new4A6000_2/tracker_rawresults/simtrack-69.3/SimTrack_VIT-B16_LaSOT/checkpoint_e500"
base_output_transt = "/home/UNT/sd1260/OSTrack_new4A6000_2/tracker_rawresults/transt-n4-64.9/LaSOT"

save_base_path = "/home/UNT/sd1260/OSTrack_new4A6000_2/assets/tracking_result_final"

# Generate full paths using the sequence name
image_folder = os.path.join(base_data_path, sequence_name, "img")
gt_bbox_path = os.path.join(base_data_path, sequence_name, "groundtruth.txt")
loretrack_bbox_path = os.path.join(base_output_loretrack, sequence_name + ".txt")
ostrack_bbox_path = os.path.join(base_output_ostrack, sequence_name + ".txt")
aiatrack_bbox_path = os.path.join(base_output_aiatrack, sequence_name + ".txt")
mixformer_bbox_path = os.path.join(base_output_mixformer, sequence_name + ".txt")
romtrack_bbox_path = os.path.join(base_output_romtrack, sequence_name + ".txt")
simtrack_bbox_path = os.path.join(base_output_simtrack, sequence_name + ".txt")
transt_bbox_path = os.path.join(base_output_transt, sequence_name + ".txt")
save_dir = os.path.join(save_base_path, sequence_name)

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Function to read bounding boxes for ground truth (comma-separated)
def read_gt_bboxes(file_path):
    bboxes = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            values = line.strip().split(",")  # Split by comma for GT
            bboxes.append(list(map(float, values)))  # Use float to handle non-integral values
    return bboxes

# Function to read bounding boxes for other trackers (space/tab-separated or comma-separated)
def read_bboxes(file_path):
    bboxes = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            
            # Detect the delimiter
            if ',' in line:
                # Handle comma-separated (e.g., ground truth)
                values = line.split(",")
            else:
                # Handle space/tab-separated (e.g., other trackers)
                values = line.split()  # Splits on any whitespace (space/tab)
            
            # Convert to float to handle non-integral values and append
            bboxes.append(list(map(float, values)))
    return bboxes

# Read bounding boxes
gt_bboxes = read_gt_bboxes(gt_bbox_path)  # Ground truth (comma-separated)
loretrack_bboxes = read_bboxes(loretrack_bbox_path)  # LoReTrack (space/tab-separated)
ostrack_bboxes = read_bboxes(ostrack_bbox_path)  # OSTrack (space/tab-separated)
aiatrack_bboxes = read_bboxes(aiatrack_bbox_path)  # AiAtrack (space/tab-separated)
mixformer_bboxes = read_bboxes(mixformer_bbox_path)  # Mixformer (space/tab-separated)
romtrack_bboxes = read_bboxes(romtrack_bbox_path)  # Romtrack (space/tab-separated)
simtrack_bboxes = read_bboxes(simtrack_bbox_path)  # Simtrack (space/tab-separated)
transt_bboxes = read_bboxes(transt_bbox_path)  # Transt (space/tab-separated)

# Get sorted list of image filenames
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])

# Ensure bbox count matches image count
assert len(image_files) == len(gt_bboxes) == len(loretrack_bboxes) == len(ostrack_bboxes) == len(aiatrack_bboxes) == len(mixformer_bboxes) == len(romtrack_bboxes) == len(simtrack_bboxes) == len(transt_bboxes), \
    "Mismatch between number of images and bounding boxes."

# Function to draw bbox without text labels
def draw_bbox(image, bbox, color):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), color, 3)  # Convert to int for drawing

# Process images
for idx, img_name in enumerate(image_files):
    img_path = os.path.join(image_folder, img_name)
    save_path = os.path.join(save_dir, img_name)

    # Load image
    image = cv2.imread(img_path)
    # draw_bbox(image, ostrack_bboxes[idx], (0, 255, 255))  # Yellow (OSTrack)
    draw_bbox(image, ostrack_bboxes[idx], (255, 105, 180))  # Pink (OSTrack)
    # draw_bbox(image, simtrack_bboxes[idx], (255, 165, 0))  # Orange (Simtrack)
    draw_bbox(image, aiatrack_bboxes[idx], (255, 0, 0))  # Blue (AiAtrack)

    draw_bbox(image, mixformer_bboxes[idx], (173, 216, 230))  # Light Blue (Mixformer)
    draw_bbox(image, romtrack_bboxes[idx], (255, 165, 0))  # Orange (Romtrack)
    draw_bbox(image, transt_bboxes[idx], (128, 128, 0))  # Olive Green (Transt)

    draw_bbox(image, gt_bboxes[idx], (0, 0, 255))  # Red (Ground truth)
    draw_bbox(image, loretrack_bboxes[idx], (0, 255, 0))  # Green (LoReTrack)
   


    # draw_bbox(image, ostrack_bboxes[idx], (0, 255, 255))  # Yellow (OSTrack)
    # draw_bbox(image, aiatrack_bboxes[idx], (255, 0, 0))  # Red (AiAtrack)
    # draw_bbox(image, mixformer_bboxes[idx], (255, 20, 147))  # Deep Pink (Mixformer)
    # draw_bbox(image, romtrack_bboxes[idx], (0, 255, 255))  # Cyan (Romtrack)
    # draw_bbox(image, simtrack_bboxes[idx], (255, 69, 0))  # Red-Orange (Simtrack)
    # draw_bbox(image, transt_bboxes[idx], (255, 105, 180))  # Hot Pink (Transt)
    # draw_bbox(image, gt_bboxes[idx], (0, 0, 255))  # Blue (Ground truth)
    # draw_bbox(image, loretrack_bboxes[idx], (0, 255, 0))  # Lime (LoReTrack)

    # Save image with bounding boxes
    cv2.imwrite(save_path, image)

    print(f"Processed: {img_name}")

print(f"All images saved in: {save_dir}")
