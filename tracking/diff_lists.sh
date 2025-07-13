#!/bin/bash

# Replace ' ' with '_' in file1.txt and save the result to file1_modified.txt
sed 's/ /_/g' /home/UNT/sd1260/OSTrack_new4A6000_2/lib/test/utils/vasttrack_test_list.txt > file1_modified.txt

# Sort the lines in both files
sort file1_modified.txt -o file1_sorted.txt
sort /home/UNT/sd1260/OSTrack_new4A6000_2/lib/test/utils/vasttrack_test_list_test.txt -o file2_sorted.txt

# Compare the differences between the two sorted files
diff file1_sorted.txt file2_sorted.txt > differences.txt

# Display the output
cat differences.txt

# Clean up temporary files if needed
# rm file1_modified.txt file1_sorted.txt file2_sorted.txt differences.txt
