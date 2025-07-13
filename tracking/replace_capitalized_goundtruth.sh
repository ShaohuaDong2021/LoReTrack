#!/bin/bash

# Path to the directory
path="/home/UNT/sd1260/OSTrack/data/vasttrack"

# Find all files named "Groundtruth.txt" in subdirectories
find "$path" -type f -name "Groundtruth.txt" | while IFS= read -r file; do
    # Rename each file to "groundtruth.txt"
    new_file=$(dirname "$file")/groundtruth.txt
    mv -v "$file" "$new_file"
done
