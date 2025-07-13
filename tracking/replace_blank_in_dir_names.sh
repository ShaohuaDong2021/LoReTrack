#!/bin/bash

# Function to replace spaces with underscores in directory names
# Run two times
replace_spaces() {
    local dir="$1"
    local new_dir="$(echo "$dir" | tr ' ' '_')"
    if [ "$dir" != "$new_dir" ]; then
        mv -v "$dir" "$new_dir"
    fi
}

# Path to the directory to be processed
path="/home/UNT/sd1260/OSTrack/data/vasttrack"

# Find all directories within the specified path
find "$path" -type d | while IFS= read -r dir; do
    replace_spaces "$dir"
done


