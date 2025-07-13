import os
import re
import matplotlib.pyplot as plt

# Path to the directory containing the result files
directory = '/home/UNT/sd1260/OSTrack_new4A6000_2/output/result_txt'

# Initialize lists to store window numbers and corresponding AUC values
window_numbers = []
auc_values = []

# Loop through files in the directory
for filename in os.listdir(directory):
    # Check if the file is a result file
    if filename.startswith("analysis_results_") and filename.endswith(".txt"):
        # Extract the window number from the filename
        match = re.search(r'analysis_results_(\d+\.\d+)\.txt', filename)
        if match:
            window_number = float(match.group(1))
            window_numbers.append(window_number)

            # Read the file and extract the AUC value
            with open(os.path.join(directory, filename), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith('OSTrack384'):
                        auc_value = float(line.split('|')[1])
                        auc_values.append(auc_value)

# Plot the AUC values
plt.figure(figsize=(10, 6))
plt.plot(window_numbers, auc_values, marker='o')
plt.xlabel('Window Number')
plt.ylabel('AUC')
plt.title('AUC Metrics vs. Window Numbers')
plt.grid(True)
plt.savefig("/home/UNT/sd1260/OSTrack_new4A6000_2/plot_figure/auc_2.pdf")
