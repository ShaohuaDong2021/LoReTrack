#!/bin/bash

cd "/home/UNT/sd1260/OSTrack_new4A6000_2/tracking/"

# Loop over the window values from 0.2 to 0.8 with a step of 0.01
for WINDOW in $(seq 0.30 0.01 0.50); do
    # Modify the config file
    sed -i "s/window:.*/window: $WINDOW/" /home/UNT/sd1260/OSTrack_new4A6000_2/experiments/ostrack/vitb_384_mae_32x4_got10k_ep100.yaml
#    sed -i "s/window:.*/window: $WINDOW/" /home/UNT/sd1260/OSTrack_new4A6000_2/experiments/ostrack/vitb_384_mae_32x4_ep300.yaml

    # Run your test command
#    python test.py ostrack vitb_384_mae_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4

    # test got-10k
    python test.py ostrack vitb_384_mae_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
    python ../lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_32x4_got10k_ep100

    # test trackingnet
#    python test.py ostrack vitb_384_mae_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
#    python ../lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name vitb_384_mae_32x4_ep300


    # Move the 'test' directory to a new directory with the window value in the name
    mv /home/UNT/sd1260/OSTrack_new4A6000_2/output/test "/home/UNT/sd1260/OSTrack_new4A6000_2/output/test_windows_$WINDOW"

    # Add a small delay between tests (optional)
    sleep 1
done

