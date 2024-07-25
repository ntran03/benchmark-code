import nibabel as nib
import os
from totalsegmentator.python_api import totalsegmentator

duke_dir = "/data/drdcad/datasets/public/DUKE_MRI_Liver/Dataset_DUKE_MRI_Liver/"
output_dir = "/data/drdcad/nicole/outputs/TS_MR/"

if __name__ == "__main__":
    # option 1: provide input and output as file paths
    #get sequences
    sequence_list = [folder for folder in os.listdir(duke_dir) if ("." not in folder and "error" not in folder and "non" not in folder)]
    sequence_list =  ["P", "Q"]
    #iterate over each sequence
    for sequence in sequence_list:
        print(f"Starting sequence {sequence}")
        file_list = [file for file in os.listdir(f"{duke_dir}{sequence}") if file.endswith(".nii.gz")]
        count = 1
        for file in file_list:
            print(f"Volume {count} out of {len(file_list)}")
            count+=1
            totalsegmentator(f"{duke_dir}{sequence}/{file}", f"{output_dir}{sequence}/{file}", task="total_mr")
    
    
    #totalsegmentator("/data/drdcad/nicole/A_SE/0001_0006.nii.gz", "/data/drdcad/nicole/outputs/TS_MR", task="total_mr")
    
    
"""     sequence = "A_nonSE"
    file_list = [file for file in os.listdir(f"{duke_dir}{sequence}") if file.endswith(".nii.gz")]
    for file in file_list:
        file_name = os.path.splitext(file)[0].split(".")[0]
        totalsegmentator(f"{duke_dir}{sequence}/{file}", f"{output_dir}{sequence}/{file_name}", task="total_mr") """
    
    
    # option 2: provide input and output as nifti image objects
    #input_img = nib.load(input_path)
    #output_img = totalsegmentator(input_img)
    #nib.save(output_img, output_path)