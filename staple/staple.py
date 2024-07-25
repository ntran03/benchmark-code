# packages
import SimpleITK as sitk # https://simpleitk.org/
import numpy as np
import os

## Some pseudo-code for you 

number = ... #the id of the image
sequence = ...

number = "0001_0032"
sequence = "B"

## read each image (TS, MRSeg, TSVIBE)
img_path = f"/data/drdcad/nicole/outputs/"
out_path = "/data/drdcad/nicole/outputs/staple"

itk_TS = sitk.ReadImage(f"{img_path}TS_MR_Complete/{sequence}/{number}")
itk_MRSeg = sitk.ReadImage(f"{img_path}MRSeg/{sequence}/{number}_seg.nii")
itk_TSVIBE = sitk.ReadImage(f"{img_path}TS_VIBE_MR/{sequence}/{number}")

l_organs = ["spleen", "right_kidney", "left_kidney", "stomach", "pancreas", "right_adrenal_gland", "left_adrenal_gland", "aorta", "inferior_vena_cava"]
## get a set of the organ IDs for the different organs in each segmentation
l_TS_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 23, 24] 
l_MRSeg_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 13, 14]
l_TSVIBE_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 25, 36]

STAPLE_seg = []
class_id = 1
## loop over the target structures of interest (in our case the list of organs I gave you before) 
for idx, organ in enumerate(l_organs):
    TS_organ_label = l_TS_organ_label_IDs[idx]
    itk_TS_organ = itk_TS == TS_organ_label
    MRSeg_organ_label = l_MRSeg_organ_label_IDs[idx]
    itk_MRSeg_organ = itk_MRSeg == MRSeg_organ_label
    TSVIBE_organ_label = l_TSVIBE_organ_label_IDs[idx]
    itk_TSVIBE_organ = itk_TSVIBE == TSVIBE_organ_label

    #convert to segmentation? also we want only the axial view
    seg1_sitk = sitk.Cast(itk_TS_organ, sitk.sitkUInt8)
    seg2_sitk = sitk.Cast(itk_MRSeg_organ, sitk.sitkUInt8)
    seg3_sitk = sitk.Cast(itk_TSVIBE_organ, sitk.sitkUInt8)

    ## combine and staple them together 
    seg_stack = [seg1_sitk, seg2_sitk, seg3_sitk]

    # Run STAPLE algorithm
    
    STAPLE_seg_sitk = sitk.STAPLE(seg_stack, class_id ) # 1.0 specifies the foreground value
    class_id+=1
    if len(STAPLE_seg)>0:
        STAPLE_seg+= sitk.GetArrayFromImage(STAPLE_seg_sitk)
    else:
        STAPLE_seg= sitk.GetArrayFromImage(STAPLE_seg_sitk)
    
    ## write stapled segmentation to disk (for your editing/correction/verification/re-annotation)
multiclass = np.sum(STAPLE_seg, axis=0)
final = sitk.GetImageFromArray(multiclass)
print(f"{out_path}/{sequence}/{number}.nii")
sitk.WriteImage(final, os.path.join(out_path, sequence, number)+".nii")