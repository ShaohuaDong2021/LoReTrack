#!/bin/bash

cd "/home/UNT/sd1260/OSTrack_new4A6000_2/tracking/"

# Loop over the window values from 0.2 to 0.8 with a step of 0.01
for WINDOW in $(seq 0.30 0.01 0.50); do
    # Modify the config file
#    scp -r /home/UNT/sd1260/OSTrack_new4A6000_2/output/test_windows_$WINDOW/tracking_results/ostrack/vitb_384_mae_32x4_got10k_ep100/got10k_submit.zip \
#    "/home/UNT/sd1260/OSTrack_new4A6000_2/output/got_window_14"
    scp -r /home/UNT/sd1260/OSTrack_new4A6000_2/output/test_windows_$WINDOW/tracking_results/ostrack/vitb_384_mae_32x4_got10k_ep100/got10k_submit.zip \
    "/home/UNT/sd1260/OSTrack_new4A6000_2/output/got_window_21"

#    cd /home/UNT/sd1260/OSTrack_new4A6000_2/output_got_needtest/got_window_11
    cd /home/UNT/sd1260/OSTrack_new4A6000_2/output/got_window_21
    mv got10k_submit.zip got10k_submit_$WINDOW.zip
    # Add a small delay between tests (optional)
    sleep 1
done

