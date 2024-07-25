#!/bin/bash

# Change directory to the desired path
cd /data/drdcad/nicole/benchmark/swarm_files

# Create an empty MRSeg_swarm.cmd file
> TSeg_swarm.cmd

# Define the path to the MRI sequences
sequencePath="/data/drdcad/datasets/public/DUKE_MRI_Liver/Dataset_DUKE_MRI_Liver/"

# Iterate over each sequence (subdirectory) in $sequencePath
for sequence in "$sequencePath"*/; do
    # Check if $sequence is a directory
    if [ -d "$sequence" ]; then
        # Iterate over each .nii.gz file in the current sequence directory
        for file in "$sequence"*.nii.gz; do
            # Extract the filename without extension
            filename=${file%.nii.gz}
            number=${filename##*/}
            sequence=${sequence%/}
            sequenceID=$(basename "$sequence")
            output_dir="/data/drdcad/nicole/outputs/TS_MR_Complete/$sequenceID/$number.nii"
            
            # Construct the mrsegmentator command and append to MRSeg_swarm.cmd
            echo "TotalSegmentator -i $file -o $output_dir --task total_mr --ml" >> TSeg_swarm.cmd
        done
    fi
done
