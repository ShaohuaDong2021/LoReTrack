#!/bin/bash

# Path to the directory
path="/home/UNT/sd1260/OSTrack/data/vasttrack"

# Find all directories named "imgs" in subdirectories
find "$path" -type d -name "imgs" | while IFS= read -r dir; do
    # Rename each directory to "img"
    new_dir=$(dirname "$dir")/img
    mv -v "$dir" "$new_dir"
done
