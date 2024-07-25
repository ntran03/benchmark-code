#!/bin/bash

# Define the paths to the two folders
folder1="/data/drdcad/datasets/public/DUKE_MRI_Liver/Dataset_DUKE_MRI_Liver/D"
folder2="/data/drdcad/nicole/outputs/MRSeg/D"

# Get list of files in both folders, considering only the first 9 characters of the filenames
files_folder1=$(cd "$folder1" && find . -type f -printf "%f\n" | cut -c 1-9 | sort -u)
files_folder2=$(cd "$folder2" && find . -type f -printf "%f\n" | cut -c 1-9 | sort -u)

# Find files that are in one folder but not in the other
echo "Files in $folder1 but not in $folder2:"
comm -23 <(echo "$files_folder1") <(echo "$files_folder2")

echo

echo "Files in $folder2 but not in $folder1:"
comm -13 <(echo "$files_folder1") <(echo "$files_folder2")
