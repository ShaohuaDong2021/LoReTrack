#!/bin/bash

mkdir "/home/UNT/sd1260/OSTrack_new4A6000_2/output/result_txt/"

cd "/home/UNT/sd1260/OSTrack_new4A6000_2/tracking/"

for WINDOW in $(seq 0.20 0.01 0.32); do

    mv "/home/UNT/sd1260/OSTrack_new4A6000_2/output/test_windows_$WINDOW" /home/UNT/sd1260/OSTrack_new4A6000_2/output/test
    # Run the analysis script and save the output to a file in the test directory
    python analysis_results.py > "/home/UNT/sd1260/OSTrack_new4A6000_2/output/result_txt/analysis_results_$WINDOW.txt"

    mv /home/UNT/sd1260/OSTrack_new4A6000_2/output/test "/home/UNT/sd1260/OSTrack_new4A6000_2/output/test_windows_$WINDOW"
done