
#!/bin/bash

cd "/home/UNT/sd1260/OSTrack_new4A6000_2/tracking/"

# Loop over the window values from 0.2 to 0.8 with a step of 0.01
for EPOCH in $(seq 291 1 300); do
    # Modify the config file
    sed -i "s/^\(  EPOCH:\).*/\1 $EPOCH/" /home/UNT/sd1260/OSTrack_new4A6000_2/experiments/ostrack/vitb_384_mae_32x4_ep300.yaml
#    sed -i "s/^\(  EPOCH:\).*/\1 $EPOCH/" /home/UNT/sd1260/OSTrack_new4A6000_2/experiments/ostrack/vitb_384_mae_32x4_got10k_ep100.yaml

    # Run your test command
    # this is the trackingnet
#    python test.py ostrack vitb_384_mae_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
#    python ../lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name vitb_384_mae_32x4_ep300

    # this is the lasot
     python test.py ostrack vitb_384_mae_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4

    # this is the got10k
#    python test.py ostrack vitb_384_mae_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
#    python ../lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_32x4_got10k_ep100
    # Move the 'test' directory to a new directory with the window value in the name
    mv /home/UNT/sd1260/OSTrack_new4A6000_2/output/test "/home/UNT/sd1260/OSTrack_new4A6000_2/output/test_got_EPOCH_$EPOCH"

    # Add a small delay between tests (optional)
    sleep 1
done


