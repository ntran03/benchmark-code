import numpy as np
import pandas as pd
import os
import SimpleITK as sitk 

## import evaluation metrics, edited some of them to remove the self parameter
from evaluation_metrics_MSD import compute_dice_coefficient, compute_surface_distances, compute_robust_hausdorff

#gt = ground truth
#cc = connected component
#np_mask_CC_GT - assume each organ has a separate label
#np_mask_CC_pred - predicted segmentation of structures/organs from TS/TSVIBE/MRSeg

#organs in the order of their labels
l_organs = ["spleen", "right_kidney", "left_kidney", "stomach", "pancreas", "right_adrenal_gland", "left_adrenal_gland", "aorta", "inferior_vena_cava", "liver"]

#sequences included in dataset
class_list =  ["arterial", "delayed", "precontrast", "T2w", "T2w_fat", "venous"]
class_list =  ["arterial", "delayed", "precontrast", "venous"]




#list of labels
l_organ_labels = [1,2,3,4,5,6,7,8,9,10]
l_TS_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 23, 24, 5] 
l_MRSeg_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 13, 14, 5]
l_TSVIBE_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 25, 36, 5]


print('#########################')
print('Computing segmentation metrics for ALL Objects - Get overlap of GT with prediction CC')



#########################################################################################################################
#Volume Computation (what's this?)

'''
## get volume for 1 voxel
unit_volume = np.prod(itk_mask_GT.GetSpacing())
## volume for GT
true_volume_voxels = np.sum(np_mask_CC_GT > 0) 
true_volume = true_volume_voxels * unit_volume * 1e-3  # for solids - cm3 or cc or cubic centimeter; for fluids - mL 

## volume for pred
pred_volume_voxels = np.sum(np_mask_CC_pred > 0)
pred_volume = pred_volume_voxels * unit_volume * 1e-3  # for solids - cm3 or cc or cubic centimeter; for fluids - mL
'''


def get_scores(input_mask_CC_GT, input_mask_CC_pred, sequence, number, segmentor):
    '''
    Does comparison of all organs in one file given input of GT and pred for file
    '''
    itk_mask_GT = sitk.ReadImage(input_mask_CC_GT)
    itk_mask_pred = sitk.ReadImage(input_mask_CC_pred)

    spacing_mask_GT = itk_mask_GT.GetSpacing()
    spacing_mm = (spacing_mask_GT[2], spacing_mask_GT[0], spacing_mask_GT[1])
    l_scores = [sequence, number] # dice and hd scores for each organ in the format [(dice, hd), (dice, hd), ...]
    #get scores per each organ
    for labelID in l_organ_labels:
        ## get organ segmentation from GT
        organ_GT = itk_mask_GT == labelID
        
        ## get organ seg from pred
        if segmentor == "ts":
            organ_pred = itk_mask_pred == l_TS_organ_label_IDs[labelID-1]
        elif segmentor == "mr":
            organ_pred = itk_mask_pred == l_MRSeg_organ_label_IDs[labelID-1]
        else:
            organ_pred = itk_mask_pred == l_TSVIBE_organ_label_IDs[labelID-1]
        ## compute dice

        #np_mask_CC_GT = sitk.Cast(organ_GT, sitk.sitkUInt16)
        np_mask_CC_GT = sitk.GetArrayFromImage(organ_GT)
        #np_mask_CC_pred = sitk.Cast(organ_pred, sitk.sitkUInt16)
        np_mask_CC_pred = sitk.GetArrayFromImage(organ_pred)
        DSC_= float(test_dice_HD(np_mask_CC_GT, np_mask_CC_pred, spacing_mm)[0])
        HD_ = float(test_dice_HD(np_mask_CC_GT, np_mask_CC_pred, spacing_mm)[1])
        print("Dice: {}, HD_95: {} mm".format(DSC_, HD_))
        #print((HD_, DSC_))
        #add organ score to list
        l_scores += [DSC_, HD_]
    return l_scores


    #return DSC_, HD_
    ## all LN
    #DSC_all, HD_all= self.test_dice_HD(np_mask_CC_GT, np_mask_CC_pred, spacing_mm)

    


#########################################################################################################################

#hausdorff distance calculation
def test_dice_HD(input_mask_CC_GT, input_mask_CC_pred, spacing_mm):

    np_mask_overlap_ = (input_mask_CC_GT > 0) * input_mask_CC_pred

    #l_overlap_idxs = np.unique(np_mask_overlap)
    l_overlap_idxs = unique2(np_mask_overlap_)
    l_overlap_idxs = np.sort(l_overlap_idxs)

    ## skip the first element, background is 0 
    l_overlap_idxs = list(l_overlap_idxs[1:])

    print('prediction labels overlap:', len(l_overlap_idxs))
    print(l_overlap_idxs)

    np_mask_pred_labels_ = np.zeros(shape = input_mask_CC_GT.shape, dtype = 'int')
    for idx in l_overlap_idxs:
        np_mask_pred_labels_[input_mask_CC_pred == idx] = 1
    
    ## compute dice
    HD_, DSC_ = compute_metrics_using_MSD_evalCode(input_mask_CC_GT > 0, np_mask_pred_labels_ > 0, spacing_mm)

    return DSC_, HD_

def compute_metrics_using_MSD_evalCode(mask_final_gt, mask_final_pred, spacing_mm):
		## calculate surface distances
		surface_distances = compute_surface_distances(mask_final_gt, mask_final_pred, spacing_mm)

		## calculate hausdorff distance
		dist_Hausdorff = compute_robust_hausdorff(surface_distances, 95)

		## calculate dice score
		dice_score = compute_dice_coefficient(mask_final_gt, mask_final_pred)

		return dist_Hausdorff, dice_score

def unique2(x):
    maxVal = np.max(x)+1
    values = np.arange(maxVal)
    used = np.zeros(maxVal)
    used[x] = 1
    return values[used==1]

def run_comparison(pred_dir, suffix, segmentor, output_path):
    '''
    Runs comparison against GT on every file in the benchmark dataset for a certain tool
    '''
    #iterate over classes
    for mri_class in class_list:
        #initialize empty results df
        df = pd.DataFrame(columns=["sequence", "number", "dsc_spleen", "hd95_spleen", "dsc_right_kidney", 
                                   "hd95_right_kidney", "dsc_left_kidney", "hd95_left_kidney", "dsc_stomach", 
                                   "hd95_stomach", "dsc_pancreas", "hd95_pancreas", "dsc_right_adrenal_gland", 
                                   "hd95_right_adrenal_gland", "dsc_left_adrenal_gland", "hd95_left_adrenal_gland", 
                                   "dsc_aorta", "hd95_aorta", "dsc_inferior_vena_cava", "hd95_inferior_vena_cava", "dsc_liver", "hd95_liver"])
        
        print(f"Starting class {mri_class}")

        file_list = [file for file in os.listdir(f"{gt_dir}{mri_class}") if file.endswith(".nii.gz")]

        for gt_file in file_list:

            #grab filename info 
            number = gt_file[-16:-7]
            sequence = gt_file[:-17]

            #grab predicted mask
            if segmentor == "mr":
                pred_file = f"{pred_dir}{sequence}/{number}_seg{suffix}"
            else:
                pred_file = f"{pred_dir}{sequence}/{number}{suffix}"
                print(pred_file)
            #pred_file = f"{gt_dir}{mri_class}/{gt_file}"

            #get dice and hd_95 (mm) scores
            scores = get_scores(f"{gt_dir}{mri_class}/{gt_file}", pred_file, sequence, number, segmentor)

            df.loc[len(df)] = scores #per organ, (dsc, hd)
            #two cols per organ, test against ground truth versus ground truth


        
        #write the class results dataframe to a csv
        df.to_csv(f"{output_path}{segmentor}/{mri_class}.csv", index=False)
        print("Written to CSV")
    

if __name__ == "__main__":
    gt_dir = "/Users/nicol/Library/CloudStorage/Box-Box/Duke_Liver_Dataset/final/" #where the benchmark masks are

    #pred_dir = "/Users/nicol/Library/CloudStorage/Box-Box/Duke_Liver_Dataset/TS_VIBE_MR/"
    #pred_dir = "/Users/nicol/Library/CloudStorage/Box-Box/Duke_Liver_Dataset/TS_MR_Complete/"
    #pred_dir = "/Users/nicol/Library/CloudStorage/Box-Box/Duke_Liver_Dataset/MRSeg/"
    #pred_dir = "/Users/nicol/Library/CloudStorage/Box-Box/Duke_Liver_Dataset/MRSeg/"
    suffix = ".nii.gz" #the filetype of the predicted masks mr
    #suffix = ".nii" #tseg, vibe

    output_path = "/Users/nicol/Documents/nih/outputs/"

    run_comparison(pred_dir, suffix, "mr", output_path)
    
##add check for the presence of the directory  "name" before starting