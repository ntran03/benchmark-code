from mrsegmentator import inference
import os


duke_dir = "/data/drdcad/datasets/public/DUKE_MRI_Liver/Dataset_DUKE_MRI_Liver/"

#duke_dir = "/data/drdcad/nicole/"
output_dir = "/data/drdcad/nicole/outputs/MRSeg/D_2"

if __name__ == "__main__":
    # option 1: provide input and output as file paths
    #get sequences
    #sequence_list = [folder for folder in os.listdir(duke_dir) if ("." not in folder and "error" not in folder)]
    #sequence_list = sequence_list[:10]
    sequence_list = ["D"]
    #iterate over each sequence
    for sequence in sequence_list:
        print(f"Starting sequence {sequence}")
        images = [f.path for f in os.scandir(f"{duke_dir}{sequence}")]
        #takes a list of image paths
        inference.infer(images, f"{output_dir}", None)
        #inference.infer(images, f"{output_dir}{sequence}")
    
    
    #totalsegmentator("/data/drdcad/nicole/A_SE/0001_0006.nii.gz", "/data/drdcad/nicole/outputs/TS_MR", task="total_mr")
    
    
"""     sequence = "A_nonSE"
    file_list = [file for file in os.listdir(f"{duke_dir}{sequence}") if file.endswith(".nii.gz")]
    for file in file_list:
        file_name = os.path.splitext(file)[0].split(".")[0]
        totalsegmentator(f"{duke_dir}{sequence}/{file}", f"{output_dir}{sequence}/{file_name}", task="total_mr") """
