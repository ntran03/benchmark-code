# packages
import SimpleITK as sitk # https://simpleitk.org/
import numpy as np
import os
import threading
import multiprocessing
import queue

## Some pseudo-code for you 

#number = ... #the id of the image
#sequence = ...

#trying to catch long runtimes
def run_staple_with_timeout(seg_stack, timeout, stop_event):
    result_queue = multiprocessing.Queue()
    stop_event = threading.Event()

    # Define inner function to execute STAPLE with timeout
    def inner_run_staple():
        nonlocal stop_event
        try:
            result = sitk.STAPLE(seg_stack, 1.0)
            if not stop_event.is_set():
                result_queue.put(result)
        except Exception as e:
            if not stop_event.is_set():
                result_queue.put(e)

    # Create and start the thread
    staple_thread = threading.Thread(target=inner_run_staple)
    staple_thread.start()

    # Wait for the thread to finish or timeout
    staple_thread.join(timeout=timeout)

    # Check if the thread is still alive
    if staple_thread.is_alive():
        print("STAPLE operation timed out.")
        stop_event.set()  # Set the stop event to signal thread termination
        return None  # Return None or handle timeout case as needed

    # Retrieve the result from the queue
    try:
        result = result_queue.get(timeout=1)  # Timeout to avoid blocking indefinitely
        if isinstance(result, sitk.Image):
            print("STAPLE operation completed successfully.")
            return result  # Return the STAPLE result
        elif isinstance(result, Exception):
            print(f"Error occurred during STAPLE operation: {result}")
            return None  # Return None or handle error case as needed
    except queue.Empty:
        print("No result received from STAPLE thread.")
        return None  # Return None or handle empty queue case as needed



def staple(number, sequence):
    file_extension = '.nii.gz'
    result_queue = multiprocessing.Queue()
    ## read each image (TS, MRSeg, TSVIBE)
    img_path = f"/data/drdcad/nicole/outputs/"
    out_path = "/data/drdcad/nicole/outputs/with_liver"

    """
    read volumes 
    1. original MRI volume
    2. segmentations (TS, MRSeg, TSVIBE)
    """

    ## --- IMPORTANT ---
    ## --- IMPORTANT ---
    ## --- IMPORTANT ---
    ## read original MRI volume also
    itk_MRI_volume = sitk.ReadImage(f"/data/drdcad/datasets/public/DUKE_MRI_Liver/Dataset_DUKE_MRI_Liver/{sequence}/{number}.nii.gz")

    ## segmentations
    itk_TS = sitk.ReadImage(f"{img_path}TS_MR_Complete/{sequence}/{number}")
    itk_MRSeg = sitk.ReadImage(f"{img_path}MRSeg/{sequence}/{number}_seg.nii")
    itk_TSVIBE = sitk.ReadImage(f"{img_path}TS_VIBE_MR/{sequence}/{number}")


    """
    extract numpy volumes from itk volumes
    1. original MRI volume
    2. segmentations (TS, MRSeg, TSVIBE)
    """

    ## MRI volume
    np_MRI_volume = sitk.GetArrayFromImage(itk_MRI_volume)

    ## segmentations
    ## numpy mask STAPLE 
    np_mask_STAPLE_final = np.zeros(np_MRI_volume.shape, dtype = 'int16')
    # ## numpy masks 
    # np_mask_TS = sitk.GetArrayFromImage(itk_TS)
    # np_mask_MRSeg = sitk.GetArrayFromImage(itk_MRSeg)
    # np_mask_TSVIBE = sitk.GetArrayFromImage(itk_TSVIBE)


    """
    make labels
    """

    l_organs = ["spleen", "right_kidney", "left_kidney", "stomach", "pancreas", "right_adrenal_gland", "left_adrenal_gland", "aorta", "inferior_vena_cava", "liver"]
    ## get a set of the organ IDs for the different organs in each segmentation
    l_TS_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 23, 24, 5] 
    l_MRSeg_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 13, 14, 5]
    l_TSVIBE_organ_label_IDs = [1, 2, 3, 6, 7, 8, 9, 25, 36, 5]


    """
    start processing
    """

    class_id = 1
    error = []

    ## loop over the target structures of interest (in our case the list of organs I gave you before) 
    for idx, organ in enumerate(l_organs):
        ## TS (use np)
        organ_label = l_TS_organ_label_IDs[idx]
        itk_TS_organ = itk_TS == organ_label
        print("completed ts segmentation")
        image_array = sitk.GetArrayFromImage(itk_TS_organ)
        num_nonzero_voxels_TS = np.count_nonzero(image_array)
        print(num_nonzero_voxels_TS)

        ## MRSeg (use np)
        organ_label = l_MRSeg_organ_label_IDs[idx]
        itk_MRSeg_organ = itk_MRSeg == organ_label
        print("completed mr segmentation")
        image_array = sitk.GetArrayFromImage(itk_MRSeg_organ)
        num_nonzero_voxels_MR = np.count_nonzero(image_array)
        print(num_nonzero_voxels_MR)


        ## TS (use np)
        organ_label = l_TSVIBE_organ_label_IDs[idx]
        itk_TSVIBE_organ = itk_TSVIBE == organ_label
        print("completed vibe segmentation")
        image_array = sitk.GetArrayFromImage(itk_TSVIBE_organ)
        num_nonzero_voxels_V = np.count_nonzero(image_array)
        print(num_nonzero_voxels_V)


        ## convert to segmentation? also we want only the axial view
        seg1_sitk = sitk.Cast(itk_TS_organ, sitk.sitkUInt16)
        seg2_sitk = sitk.Cast(itk_MRSeg_organ, sitk.sitkUInt16)
        seg3_sitk = sitk.Cast(itk_TSVIBE_organ, sitk.sitkUInt16)
        print("completed conversion")

        

        #ssh tranne@biowulf.nih.gov
        #cd /data/drdcad/nicole/benchmark
        if (num_nonzero_voxels_TS < 1 and num_nonzero_voxels_MR < 1 and num_nonzero_voxels_V < 1):
            print(f"{organ} is not there")
            #error.append(organ)
        else:
            ## combine and staple them together 
            seg_stack = [seg1_sitk, seg2_sitk, seg3_sitk]

            # ## Run STAPLE algorithm    
            # STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0) # 1.0 specifies the foreground value
            # print("completed staple")
            
            result = run_staple_with_timeout(seg_stack, timeout=600, stop_event=threading.Event())
            if result is not None:
                STAPLE_seg_sitk = result
                 ## get numpy mask
                np_mask_STAPLE = sitk.GetArrayFromImage(STAPLE_seg_sitk)
                print("converted to numpy")
                #print(np_mask_STAPLE)

                ## set final STAPLE mask to class ID
                np_mask_STAPLE_final[np_mask_STAPLE >= 0.01] = class_id
                print("added to final stack")
                #print(np_mask_STAPLE_final)
            else:
                print("failed")
                error.append(organ)
                pass

            ## increment class ID
        class_id += 1
        print(organ)
    """
    convert final numpy mask to ITK image 
    """

    ## convert mask as ITK image
    itk_mask_final = sitk.GetImageFromArray(np_mask_STAPLE_final.astype('int16'))
    #sitk.WriteImage(itk_mask_final, "testfinal.nii")

    ## --- IMPORTANT ---
    ## --- IMPORTANT ---
    ## --- IMPORTANT ---
    ## copy over DICOM/NIFTI header information from reference volume (let's say MRI/CT volume/mask)
    ## the final volume should have the same volume dimensions + origin/spacing/direction as the reference volume (MRI volume)
    itk_mask_final.CopyInformation(itk_MRI_volume)

    """
    save segmentation to disk 
    ## write stapled segmentation to disk (for your editing/correction/verification/re-annotation)
    """

    print(f"{out_path}/{sequence}/{number}.nii")

    sitk.WriteImage(itk_mask_final, os.path.join(out_path, f"{sequence}_{number}")+".nii")
    print(error)
    if len(error) > 0:
        print(error)
        #with open("/data/drdcad/nicole/benchmark/error.txt", "a") as f:
        #    f.write(f"{sequence} {number} {error} \n")
    

if __name__ == "__main__":
    #sequence_list =  ["A_SE", "A_SE_T2w", "B", "C", "E", "K", "O", "Q"]
    #duke_dir = "/data/drdcad/datasets/public/DUKE_MRI_Liver/Dataset_DUKE_MRI_Liver/"
    staple_dir = "/data/drdcad/nicole/outputs/final/"
    #iterate over each sequence
    #staple(number="0104_0012", sequence="A_SE_T2w")
    class_list = ["T2w", "T2w_fat", "venous"]

    for class_name in class_list:
        print(f"Starting class {class_name}")
        file_list = [file for file in os.listdir(f"{staple_dir}{class_name}") if file.endswith(".nii")]
        count = 1
        for file in file_list:
            print(f"Volume {count} out of {len(file_list)}")
            count+=1
            staple(number=file[-13:-4], sequence = file[:-14])
    '''
    for sequence in sequence_list:
        print(f"Starting sequence {sequence}")
        file_list = [file for file in os.listdir(f"{duke_dir}{sequence}") if file.endswith(".nii.gz")]
        count = 1
        for file in file_list:
            print(f"Volume {count} out of {len(file_list)}")
            count+=1
            staple(number=file.split(".")[0], sequence=sequence)
    '''