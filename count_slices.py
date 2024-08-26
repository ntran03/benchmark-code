import nibabel as nib
import os

folder_path = '/Users/nicol/Library/CloudStorage/Box-Box/Duke_Liver_Dataset/final/venous/images'
total_slices = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.nii') or filename.endswith('.nii.gz'):
        filepath = os.path.join(folder_path, filename)
        img = nib.load(filepath)
        shape = img.shape
        num_slices = shape[2]  # Assuming axial slices are the 3rd dimension
        total_slices += num_slices
        print(f"{filename}: {num_slices} slices")



print(f"Total slices in all volumes: {total_slices}")
