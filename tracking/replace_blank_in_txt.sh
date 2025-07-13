#!/bin/bash

# Path to the directory
path="/home/UNT/sd1260/OSTrack/data/vasttrack"

# Name of the text file to store the directory names
output_file="/home/UNT/sd1260/OSTrack_new4A6000_2/lib/test/utils/vasttrack_test_list_test.txt"

# Find all subdirectories (max depth is 1, excluding the current directory)
find "$path" -mindepth 2 -maxdepth 2 -type d -exec basename {} \; > "$output_file"

echo "Subdirectories listed in $output_file:"
cat "$output_file"
