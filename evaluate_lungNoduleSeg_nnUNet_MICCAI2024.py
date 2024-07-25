"""
Evaluation code to calculate metrics for nnUNet executed on test data 
For MICCAI 2024 
Written by TSM - 21 June 2024
- nnUNet was trained on unitochest dataset (all patients with single studies)
- nnUNet evaluated on unitochest dataset (all patients with multiple studies) 
- Model was trained with 29 structures (extracted with TotalSegmentator run on unitochest dataset)
"""

import os 
import random 
import glob
from pathlib import Path

import numpy as np 
import scipy as sp
import pandas as pd

import math 

from enum import Enum

import matplotlib.pyplot as plt

from itertools import chain

# import nnunet 
import SimpleITK as sitk 
import nibabel as nib
from scipy.spatial import distance
from skimage.measure import label, regionprops, regionprops_table
import feret

## import evaluation metrics 
from evaluation_metrics_MSD import compute_dice_coefficient, compute_surface_distances, compute_robust_hausdorff


class LNCT_evaluation:
	
	def __init__(self,
					dir_mask_GT = '',
					dir_mask_pred = ''
				):
		
		## data locations
		self.dir_mask_GT = dir_mask_GT
		self.dir_mask_pred = dir_mask_pred

		## list of files 
		self.l_ffpn_masks_GT = []
		self.l_ffpn_masks_pred = []

	#########################################################################################################################
	
	def get_list_filesInFolder(self, dir_data):
		
		## get list of nifti masks in folder
		l_ffpn_masks = []
		for f in sorted(os.listdir(dir_data)):	
			if os.path.isfile(os.path.join(dir_data, f)) and f.endswith('.nii.gz'):
					ffpn_mask = os.path.join(dir_data, f)
					l_ffpn_masks.append(ffpn_mask)

		## sort files in ascending order 					
		l_ffpn_masks = sorted(l_ffpn_masks, key = lambda x: int(os.path.basename(x).split('.nii.gz')[0].split('_')[0]))

		return l_ffpn_masks
	
	#########################################################################################################################
	
	def getConnectedComponents_simpleITK(self, vol):
		"""
		https://discourse.itk.org/t/simpleitk-extract-largest-connected-component-from-binary-image/4958
		"""

		# 1. Convert binary image into a connected component image, each component has an integer label.
		# 2. Relabel components so that they are sorted according to size (there is an
		#    optional minimumObjectSize parameter to get rid of small components).
		# 3. Get largest connected componet, label==1 in sorted component image.
		component_image = sitk.ConnectedComponent(vol)
		sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
		#largest_component_binary_image = sorted_component_image == 1

		return sorted_component_image

	#########################################################################################################################
	
	def compute_metrics_using_MSD_evalCode(self, mask_final_gt, mask_final_pred, spacing_mm):
		## calculate surface distances
		surface_distances = compute_surface_distances(mask_final_gt, mask_final_pred, spacing_mm)
		print(surface_distances)
		## calculate hausdorff distance
		dist_Hausdorff = compute_robust_hausdorff(surface_distances, 95)

		## calculate dice score
		dice_score = compute_dice_coefficient(mask_final_gt, mask_final_pred)

		return dist_Hausdorff, dice_score
	
	#########################################################################################################################
	
	def test_dice_HD(self, input_mask_CC_GT, input_mask_CC_pred, spacing_mm):
				
		np_mask_overlap_ = (input_mask_CC_GT > 0) * input_mask_CC_pred

		#l_overlap_idxs = np.unique(np_mask_overlap)
		l_overlap_idxs = self.unique2(np_mask_overlap_)
		l_overlap_idxs = np.sort(l_overlap_idxs)

		## skip the first element, background is 0 
		l_overlap_idxs = list(l_overlap_idxs[1:])
		
		print('prediction labels overlap:', len(l_overlap_idxs))
		print(l_overlap_idxs)

		np_mask_pred_labels_ = np.zeros(shape = input_mask_CC_GT.shape, dtype = 'int')
		for idx in l_overlap_idxs:
			np_mask_pred_labels_[input_mask_CC_pred == idx] = 1

		## compute dice
		HD_, DSC_ = self.compute_metrics_using_MSD_evalCode(
																input_mask_CC_GT > 0, 
																np_mask_pred_labels_ > 0, 
																spacing_mm
															)

		## simple dice
		simple_HD, simple_DSC = self.compute_metrics_using_MSD_evalCode(
																			input_mask_CC_GT > 0, 
																			np_mask_overlap_ > 0, 
																			spacing_mm
																		)
		
		return DSC_, HD_, simple_DSC, simple_HD
	
	#########################################################################################################################
	
	def unique2(self, x):
		maxVal    = np.max(x)+1
		values    = np.arange(maxVal)
		used      = np.zeros(maxVal)
		used[x]   = 1
		return values[used==1]
	
	#########################################################################################################################

	def dice_coefficient(self, y_true, y_pred):
		return 1 - distance.dice(y_true.flatten(), y_pred.flatten())
	
	#########################################################################################################################
	
	def set_mask_value(self, image, mask, value):
				msk32 = sitk.Cast(mask, sitk.sitkFloat32)
				return sitk.Cast(sitk.Cast(image, sitk.sitkFloat32) *
								sitk.InvertIntensity(msk32, maximum=1.0) + 
								msk32*value, image.GetPixelID())
	
	#########################################################################################################################

	def stratify_objects_by_size_GT(self, np_mask_GT, uniqueLabels_GT, spacing_mask_GT):

		"""
		For FASTER processing, precompute GT LN above/below Xmm size - this is GT labels above/below Xmm
		Use commented code below if needed, then copy the precomputed GT LN of size into the variables below
		"""
		
		# l_GT_labels_lesserThan_Xcm = [
		# 	[1, 2, 3, 4, 5, 6, 7, 9, 10, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 38, 41, 42, 43, 44],
		# 	[1, 2, 3, 6, 9, 10, 15, 16, 17, 18, 19, 20],
		# 	[1, 3, 5, 6, 7, 8],
		# 	[1, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 27],
		# 	[1, 2, 3, 4, 5, 10],
		# 	[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 17, 18, 19, 20, 21, 22],
		# 	[1, 2, 10, 11, 14, 15, 16, 17, 18, 19],
		# 	[1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 17, 18, 19, 20, 21, 27, 29, 33, 36, 43, 44, 45],
		# 	[1, 2, 3, 4, 5, 6, 9, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 44],
		# 	[1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 15, 16, 17, 20, 23, 25, 26, 27, 28, 31],
		# 	[3, 6, 7, 8, 9, 10],
		# 	[2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13],
		# 	[1, 2, 3, 5, 8, 9, 11, 12, 15, 21, 22, 23],
		# 	[1, 2, 3, 6, 10, 13, 17, 18, 19, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32],
		# 	[1, 3, 4, 6, 7, 8, 9, 12, 15, 16],

		# ]

		# l_GT_labels_greaterThan_Xcm = [
		# 	[8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 27, 28, 29, 30, 31, 32, 37, 39, 40],
		# 	[4, 5, 7, 8, 11, 12, 13, 14],
		# 	[2, 4],
		# 	[2, 6, 18, 19, 20, 22, 23, 24, 25, 26],
		# 	[6, 7, 8, 9],
		# 	[11, 13, 16],
		# 	[3, 4, 5, 6, 7, 8, 9, 12, 13],
		# 	[7, 9, 10, 12, 15, 16, 22, 23, 24, 25, 26, 28, 30, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 46, 47, 48, 49, 50],
		# 	[7, 8, 10, 11, 12, 13, 14, 20, 28, 30, 35, 43],
		# 	[7, 10, 13, 14, 18, 19, 21, 22, 24, 29, 30],
		# 	[1, 2, 4, 5],
		# 	[1, 4],
		# 	[4, 6, 7, 10, 13, 14, 16, 17, 18, 19, 20],
		# 	[4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 20, 24, 26],
		# 	[2, 5, 10, 11, 13, 14]

		# ]

		"""
		#########################################################################################################################
		Ground Truth - Differentiate small from large objects based on length of major/minor Feret diameter
		90* normal Feret diameter is perpendicular to the max Feret Diameter 
		The guidelines change by type of finding.
		
		Lymph nodes - short axis diameter (minor Feret) >= 8mm is clinically significant, >= 1cm is suspicious for metastasis
		
		Nodules - 2017 Fleishner society guidelines (https://radiopaedia.org/articles/fleischner-society-pulmonary-nodule-recommendations-1?lang=us)
			- small nodules 3-10 mm should be expressed, for risk estimation purposes, as the average of the short-axis and long-axis diameters (measured on the same slice)
			- small nodules <3 mm should not be measured and should be described as micronodules
			- larger nodules >10 mm and masses, for descriptive purposes, should be described in both short- and long-axis measurements
		"""

		print('Partitioning GT by size:')

		np.set_printoptions(precision=3)

		l_GT_labels_greaterThan_Xcm = []
		l_GT_labels_lesserThan_Xcm = []
		# np_mask_all_GT_labels_greaterThan_Xmm = np.zeros(shape = np_mask_GT.shape, dtype = 'int')
		# np_mask_all_GT_labels_lesserThan_Xmm = np.zeros(shape = np_mask_GT.shape, dtype = 'int')

		d_GT_label_attributes = {}

		for labelIdx in uniqueLabels_GT:

			#print('==')
			#print('labelIdx:', labelIdx)

			"""
			for current label ID, find slice in 3D volume with maximum extent (max num pixels)
			"""

			## get mask for current label
			np_mask_GT_label = (np_mask_GT == labelIdx) * 1 
			#print('shape:', np_mask_GT_label.shape)
			## get number of pixels in each slice
			l_label_numPixelsPerSlice = np.sum(np_mask_GT_label, axis = (1,2)) ## sum over each slice
			## get max val + idx 
			maxVal = np.max(l_label_numPixelsPerSlice)
			maxValIdx = np.argmax(l_label_numPixelsPerSlice)
			#print('max:', l_label_numPixelsPerSlice[maxValIdx], maxVal, maxValIdx)
			## get a mask of slices with label in them 
			l_label_slicesMask = l_label_numPixelsPerSlice > 0 
			## find slice indicies with label in it
			l_nonzeroIdxs = [idx for idx, v in enumerate(l_label_slicesMask) if v != 0]
			# ## min/max slice index with label in it
			# sliceIdx_labelInIt_min = l_nonzeroIdxs[0]
			# sliceIdx_labelInIt_max = l_nonzeroIdxs[-1]

			## create entry for GT label
			d_GT_label_attributes[labelIdx] = {}
			## store GT label attributes 
			d_GT_label_attributes[labelIdx]['l_label_numPixelsPerSlice'] = l_label_numPixelsPerSlice ## sum of voxels in each slice for object
			d_GT_label_attributes[labelIdx]['sliceIdxWithMaxVoxels'] = maxValIdx ## slice idx with maximum number of voxels for object 
			d_GT_label_attributes[labelIdx]['maxVoxelCount'] = maxVal ## number (max) of voxels for object 
			# d_GT_label_attributes[labelIdx]['sliceIdx_labelInIt_min'] = sliceIdx_labelInIt_min ## slice idx denoting minimum extent of object in volume
			# d_GT_label_attributes[labelIdx]['sliceIdx_labelInIt_max'] = sliceIdx_labelInIt_max ## slice idx denoting max extent of object in volume
			
			"""
			for current label ID, extract slice with max extent 
			"""

			## get slice with max volume 
			mask_slice_maxVol = np_mask_GT_label[maxValIdx, :, :]

			## assert spacing in X == Y
			assert spacing_mask_GT[0] == spacing_mask_GT[1]
			
			"""
			Compute size of the object (Feret diameters, major + minor)

			## here, we have slice with largest extent of object
			## now, we can now get the object size based on its Feret diameter 

			## for lymph nodes, size should be perpendicular to the Feret diameter (to get the short axis diameter)
			## get only maxferet90 distance (length of line perpendicular to the maximum Feret diameter)

			## for lung nodules, size is the average nodule diameter between the long and the short axis in whichever plane (axial, coronal or sagittal) the nodule shows its maximum dimension
			"""
			# # maxf90 = feret.max90(mask_slice_maxVol)
			# get all the values
			maxf, minf, minf90, maxf90 = feret.all(mask_slice_maxVol)

			# ## plot output -- patient 1, LN ID 17
			# if labelIdx == 17:
			# 	# plot the result
			# 	feret.plot(mask_slice_maxVol)

			# 	plt.savefig('test.png', bbox_inches='tight')
			# 	plt.close()

			# # get all the informations
			# res = feret.calc(mask_slice_maxVol)
			# maxf = res.maxf
			# minf =  res.minf
			# minf90 = res.minf90
			# minf_angle = res.minf_angle
			# minf90_angle = res.minf90_angle
			# maxf_angle = res.maxf_angle
			# maxf90_angle = res.maxf90_angle			
			
			## multiply spacing to distance 
			v_majorAxisLength_mm = maxf * spacing_mask_GT[0]
			v_minorAxisLength_mm = maxf90 * spacing_mask_GT[0]

			## average diameter
			v_averageDiameter_mm = (v_majorAxisLength_mm + v_minorAxisLength_mm) / 2

			print(
					'maxf:', round(maxf, 2), 
					'\t maxf90:', round(maxf90, 2),
					'\t avgD_mm:', round(v_averageDiameter_mm, 2), 
					'\t majorD_mm:', round(v_majorAxisLength_mm, 2),
					'\t minorD_mm:', round(v_minorAxisLength_mm, 2)
				)	

			#print('v_minorAxisLength_mm:', v_minorAxisLength_mm)

			# """
			# ## for lymph nodes, size should be perpendicular to the Feret diameter (to get the short axis diameter)
			# ## get only maxferet90 distance (length of line perpendicular to the maximum Feret diameter)
			# """
			# # ## greater than 1cm
			# # if v_minorAxisLength_mm > 10:
			# ## greater than 8mm
			# if v_minorAxisLength_mm >= 8:
			# 	l_GT_labels_greaterThan_Xcm.append(labelIdx)
			# 	np_mask_all_GT_labels_greaterThan_Xmm[np_mask_GT == labelIdx] = labelIdx
			# else:
			# 	l_GT_labels_lesserThan_Xcm.append(labelIdx)
			# 	np_mask_all_GT_labels_lesserThan_Xmm[np_mask_GT == labelIdx] = labelIdx


			"""
			Differentiate object by size 

			## for lung nodules

			In the attempt to reduce variability in nodule measurements, the latest version of the Fleischner Society guidelines published in 2017 
			recommended the calculation of the average nodule diameter between the long and the short axis in whichever plane (axial, coronal or sagittal) the nodule shows its maximum dimension [7]. 
			A following statement focused on recommendations for measuring pulmonary nodules clarified that for nodules < 1cm the dimension should be expressed as average diameter, 
			while for larger nodules both short- and long-axis diameters taken on the same plane should be reported [44].
			"""
			## greater than 1cm			
			if v_averageDiameter_mm >= 10:
				l_GT_labels_greaterThan_Xcm.append(labelIdx)
				#np_mask_all_GT_labels_greaterThan_Xmm[np_mask_GT == labelIdx] = labelIdx
			else:
				l_GT_labels_lesserThan_Xcm.append(labelIdx)
				#np_mask_all_GT_labels_lesserThan_Xmm[np_mask_GT == labelIdx] = labelIdx

		print('less than Xcm:', len(l_GT_labels_lesserThan_Xcm))
		print(l_GT_labels_lesserThan_Xcm)
		print('greater than Xcm:', len(l_GT_labels_greaterThan_Xcm))
		print(l_GT_labels_greaterThan_Xcm)

		# ## add to counter 
		# num_LN_in_GT_greaterThan_Xcm += len(l_GT_labels_greaterThan_Xcm)
		# num_LN_in_GT_lesserThan_Xcm += len(l_GT_labels_lesserThan_Xcm)

		# print('np_mask_all_GT_labels_lesserThan_Xmm.shape:', np_mask_all_GT_labels_lesserThan_Xmm.shape)
		# print('np_mask_all_GT_labels_greaterThan_Xmm.shape:', np_mask_all_GT_labels_greaterThan_Xmm.shape)

		return l_GT_labels_lesserThan_Xcm, l_GT_labels_greaterThan_Xcm
	
	#########################################################################################################################

	def stratify_objects_by_size_prediction(self, np_mask_CC_pred, uniqueLabels_pred, spacing_mask_pred):

		l_pred_labels_lesserThan_Xcm = []
		l_pred_labels_greaterThan_Xcm = []			
		
		# np_mask_all_GT_labels_lesserThan_Xmm = np.zeros(shape = np_mask_GT.shape, dtype = 'int')
		# np_mask_all_GT_labels_greaterThan_Xmm = np.zeros(shape = np_mask_GT.shape, dtype = 'int')

		# d_GT_label_attributes = {}

		np.set_printoptions(precision=3)

		for labelIdx in uniqueLabels_pred:

			#print('==')
			#print('labelIdx:', labelIdx)

			"""
			for current label ID, find slice in 3D volume with maximum extent (max num pixels)
			"""

			## get mask for current label
			np_mask_pred_label = (np_mask_CC_pred == labelIdx) * 1 
			#print('shape:', np_mask_GT_label.shape)
			## get number of pixels in each slice
			l_label_numPixelsPerSlice = np.sum(np_mask_pred_label, axis = (1,2)) ## sum over each slice
			## get max val + idx 
			maxVal = np.max(l_label_numPixelsPerSlice)
			maxValIdx = np.argmax(l_label_numPixelsPerSlice)
			#print('max:', l_label_numPixelsPerSlice[maxValIdx], maxVal, maxValIdx)
			## get a mask of slices with label in them 
			l_label_slicesMask = l_label_numPixelsPerSlice > 0 
			## find slice indicies with label in it
			l_nonzeroIdxs = [idx for idx, v in enumerate(l_label_slicesMask) if v != 0]
			# ## min/max slice index with label in it
			# sliceIdx_labelInIt_min = l_nonzeroIdxs[0]
			# sliceIdx_labelInIt_max = l_nonzeroIdxs[-1]
			
			"""
			for current label ID, extract slice with max extent 
			"""

			## get slice with max volume 
			mask_slice_maxVol = np_mask_pred_label[maxValIdx, :, :]

			## assert spacing in X == Y
			assert spacing_mask_pred[0] == spacing_mask_pred[1]

			"""
			Compute size of the object (Feret diameters, major + minor)

			## here, we have slice with largest extent of object
			## now, we can now get the object size based on its Feret diameter 

			## for lymph nodes, size should be perpendicular to the Feret diameter (to get the short axis diameter)
			## get only maxferet90 distance (length of line perpendicular to the maximum Feret diameter)

			## for lung nodules, size is the average nodule diameter between the long and the short axis in whichever plane (axial, coronal or sagittal) the nodule shows its maximum dimension
			"""
			# # maxf90 = feret.max90(mask_slice_maxVol)
			# get all the values
			maxf, minf, minf90, maxf90 = feret.all(mask_slice_maxVol)

			# ## plot output -- patient 1, LN ID 17
			# if labelIdx == 17:
			# 	# plot the result
			# 	feret.plot(mask_slice_maxVol)

			# 	plt.savefig('test.png', bbox_inches='tight')
			# 	plt.close()

			# # get all the informations
			# res = feret.calc(mask_slice_maxVol)
			# maxf = res.maxf
			# minf =  res.minf
			# minf90 = res.minf90
			# minf_angle = res.minf_angle
			# minf90_angle = res.minf90_angle
			# maxf_angle = res.maxf_angle
			# maxf90_angle = res.maxf90_angle			
			
			## multiply spacing to distance 
			v_majorAxisLength_mm = maxf * spacing_mask_pred[0]
			v_minorAxisLength_mm = maxf90 * spacing_mask_pred[0]

			## average diameter
			v_averageDiameter_mm = (v_majorAxisLength_mm + v_minorAxisLength_mm) / 2

			print(
					'maxf:', round(maxf, 2), 
					'\t maxf90:', round(maxf90, 2),
					'\t avgD_mm:', round(v_averageDiameter_mm, 2), 
					'\t majorD_mm:', round(v_majorAxisLength_mm, 2),
					'\t minorD_mm:', round(v_minorAxisLength_mm, 2)
				)	

			#print('v_minorAxisLength_mm:', v_minorAxisLength_mm)


			"""
			Differentiate object by size 

			## for lung nodules

			In the attempt to reduce variability in nodule measurements, the latest version of the Fleischner Society guidelines published in 2017 
			recommended the calculation of the average nodule diameter between the long and the short axis in whichever plane (axial, coronal or sagittal) the nodule shows its maximum dimension [7]. 
			A following statement focused on recommendations for measuring pulmonary nodules clarified that for nodules < 1cm the dimension should be expressed as average diameter, 
			while for larger nodules both short- and long-axis diameters taken on the same plane should be reported [44].
			"""
			## greater than 1cm			
			if v_averageDiameter_mm >= 10:
				l_pred_labels_greaterThan_Xcm.append(labelIdx)
				#np_mask_all_GT_labels_greaterThan_Xmm[np_mask_GT == labelIdx] = labelIdx
			else:
				l_pred_labels_lesserThan_Xcm.append(labelIdx)
				#np_mask_all_GT_labels_lesserThan_Xmm[np_mask_GT == labelIdx] = labelIdx

		return l_pred_labels_lesserThan_Xcm, l_pred_labels_greaterThan_Xcm

	#########################################################################################################################

	def collate_results_(self, l_cases, d_dice, d_HD, d_volumeCalc):

		d_final_results = {}

		## store test patient volume names
		d_final_results['case'] = l_cases

		## store dice metrics for test cases
		for l_str in d_dice.keys():
			d_final_results[l_str] = d_dice[l_str]

		## store HD metrics for test cases
		for l_str in d_HD.keys():
			d_final_results[l_str] = d_HD[l_str]

		## store volume metrics for test cases
		for l_str in d_volumeCalc.keys():
			d_final_results[l_str] = d_volumeCalc[l_str]

		# ## store detection metrics for test cases
		# for l_str in d_detectionMetrics.keys():
		# 	d_final_results[l_str] = d_detectionMetrics[l_str]

		## convert dict to df 
		df_results = pd.DataFrame.from_dict(d_final_results, orient='index').transpose()

		print('final_df shape:', df_results.shape)

		return df_results
	
	#########################################################################################################################

	def dump_results_to_csv(self, df_results, str_fn_2_save):

		## make csv filename to save
		fn_2_save = str_fn_2_save + '.csv'

		print(fn_2_save)

		## save df to disk 
		df_results.to_csv(fn_2_save, encoding='utf-8', index=False)
		
	#########################################################################################################################
	
	def compute_metrics_simpleITK(self, l_GT, l_pred):
		
		num_patients = len(l_GT)

		## get number of ROI in all GT
		num_ROI_in_all_GT = 0

		## get shape of all CT volumes 
		l_all_GT_shape = []

		l_cases = []

		total_GT = 0 
		total_GT_greaterThan_Xmm = 0
		total_tp = 0
		total_fp = 0
		total_fn = 0

		## store detection metrics
		d_detectionMetrics = {}
		## store  
		d_detectionMetrics['l_GT_all'] = []
		d_detectionMetrics['l_numGT_all'] = []
		d_detectionMetrics['l_GT_TP_all'] = []
		d_detectionMetrics['l_tp_all'] = []
		d_detectionMetrics['l_fp_all'] = []
		d_detectionMetrics['l_fn_all'] = []
		## store  
		d_detectionMetrics['l_GT_lesserThan_Xmm'] = []
		d_detectionMetrics['l_numGT_lesserThan_Xmm'] = []
		d_detectionMetrics['l_GT_TP_lesserThan_Xmm'] = []
		d_detectionMetrics['l_tp_lesserThan_Xmm'] = []
		d_detectionMetrics['l_fp_lesserThan_Xmm'] = []
		d_detectionMetrics['l_fn_lesserThan_Xmm'] = []
		## store  
		d_detectionMetrics['l_GT_greaterThan_Xmm'] = []
		d_detectionMetrics['l_numGT_greaterThan_Xmm'] = []
		d_detectionMetrics['l_GT_TP_greaterThan_Xmm'] = []
		d_detectionMetrics['l_tp_greaterThan_Xmm'] = []
		d_detectionMetrics['l_fp_greaterThan_Xmm'] = []
		d_detectionMetrics['l_fn_greaterThan_Xmm'] = []

		## store Hausdorff Distance
		d_HD95 = {}
		d_HD95['l_HD_all'] = []
		d_HD95['l_HD_lesserThan_Xmm'] = []
		d_HD95['l_HD_greaterThan_Xmm'] = []
		d_HD95['l_HD_all_simple'] = []
		d_HD95['l_HD_lesserThan_Xmm_simple'] = []
		d_HD95['l_HD_greaterThan_Xmm_simple'] = []

		## store Dice score
		d_DSC = {}
		d_DSC['l_DSC_all'] = []
		d_DSC['l_DSC_lesserThan_Xmm'] = []
		d_DSC['l_DSC_greaterThan_Xmm'] = []
		d_DSC['l_DSC_all_simple'] = []
		d_DSC['l_DSC_lesserThan_Xmm_simple'] = []
		d_DSC['l_DSC_greaterThan_Xmm_simple'] = []

		## store volume
		d_volumeCalc = {}
		d_volumeCalc['l_vol_true_voxels'] = []
		d_volumeCalc['l_vol_true_cc'] = []
		d_volumeCalc['l_vol_pred_voxels'] = []
		d_volumeCalc['l_vol_pred_cc'] = []
		d_volumeCalc['l_unit_volume'] = []


		l_test_dice = []

		## set output print options 
		np.set_printoptions(precision=3)

		for curr_study_idx in range(num_patients):
		# for curr_study_idx in range(5):

			print('============' * 10)
			print('============' * 10)
			print('============' * 10)
			print('curr_study_idx:', curr_study_idx + 1)

			

			"""
			#########################################################################################################################
			Initialize metrics container
			"""

			curr_vol_tp = 0
			curr_vol_fp = 0
			curr_vol_fn = 0			

			"""
			#########################################################################################################################
			Get filenames
			"""			

			## get GT
			ffpn_GT = l_GT[curr_study_idx]
			## get pred
			ffpn_pred = l_pred[curr_study_idx]
			
			## get study name 
			fn_study = os.path.basename(ffpn_GT)
			print('study:', fn_study)

			## store study
			l_cases.append(fn_study)

			"""
			#########################################################################################################################
			Read GT + prediction
			"""

			## read GT
			print('Reading mask GT:')
			itk_mask_GT = sitk.ReadImage(ffpn_GT)
			## read pred
			print('Reading mask pred:')
			itk_mask_pred = sitk.ReadImage(ffpn_pred)

			## get size of volume 
			t_CT_size = itk_mask_GT.GetSize()
			## store
			l_all_GT_shape.append(t_CT_size)

			## get dimensions
			ttt = np.asarray(l_all_GT_shape)
			for colSizeIdx in range(3):
				print('shape -- min, max:', np.min(ttt[:,colSizeIdx]), np.max(ttt[:,colSizeIdx]))

			## get spacing (x/y/z voxel resolution)
			spacing_mask_GT = itk_mask_GT.GetSpacing()
			print('spacing_mask_GT:', spacing_mask_GT)
			spacing_mask_pred = itk_mask_pred.GetSpacing()
			print('spacing_mask_pred:', spacing_mask_pred)			

			"""
			#########################################################################################################################
			Process Prediction
			"""			

			np_temp_mask_pred = sitk.GetArrayFromImage(itk_mask_pred)
			
			#itk_temp_pred_connectedComponents = sitk.ConnectedComponent(sitk.GetImageFromArray(np_actual_mask_pred))

			print('Running ITK connected components on ROI predictions')
			
			### Connected components 
			## ablation experiment (ROI without anatomy priors)
			if 'ablationExpOnly' in self.dir_mask_pred:
				## take all ROI 
				itk_pred_connectedComponents = sitk.ConnectedComponent(itk_mask_pred > 0)
			## regular (ROI with anatomy priors)
			else:
				## ROI label is almost always label 2, extract that
				itk_pred_connectedComponents = sitk.ConnectedComponent(itk_mask_pred == 2)

			## numpy pred
			np_mask_CC_pred = sitk.GetArrayFromImage(itk_pred_connectedComponents)

			print('np_mask_CC_pred.shape:', np_mask_CC_pred.shape)

			# these are the unique labels in the prediction (including TS labels and/or true seg labels)
			uniqueLabels_pred = self.unique2(np_mask_CC_pred)
			uniqueLabels_pred = np.sort(uniqueLabels_pred)
			## skip the first element, background is 0 
			uniqueLabels_pred = uniqueLabels_pred[1:]
			print('uniqueLabels_pred (only ROI):', len(uniqueLabels_pred))
			print(uniqueLabels_pred)

			print('filtering small predictions that may be incorrect')

			## filter small predictions
			for pred_label in uniqueLabels_pred:
				np_mask_pred_label = np_mask_CC_pred == pred_label
				if np.sum(np_mask_pred_label) <= 10:
					itk_pred_connectedComponents = self.set_mask_value(itk_pred_connectedComponents, itk_pred_connectedComponents == pred_label, 0)

			## redo the CCA 
			itk_pred_connectedComponents = sitk.ConnectedComponent(itk_pred_connectedComponents > 0)

			## numpy pred
			np_mask_CC_pred = sitk.GetArrayFromImage(itk_pred_connectedComponents)

			print('np_mask_CC_pred.shape:', np_mask_CC_pred.shape)

			# these are the unique labels in the prediction (including TS labels and/or true seg labels)
			uniqueLabels_pred = self.unique2(np_mask_CC_pred)
			uniqueLabels_pred = np.sort(uniqueLabels_pred)
			## skip the first element, background is 0 
			uniqueLabels_pred = uniqueLabels_pred[1:]
			print('uniqueLabels_pred (only ROI):', len(uniqueLabels_pred))
			print(uniqueLabels_pred)


			"""
			#########################################################################################################################
			Process GT
			"""

			## take all ROI (anything > 0)
			itk_GT_connectedComponents = sitk.ConnectedComponent(itk_mask_GT > 0)

			## numpy GT 
			np_mask_CC_GT = sitk.GetArrayFromImage(itk_GT_connectedComponents)

			print('np_mask_CC_GT.shape:', np_mask_CC_GT.shape)

			## find unique labels in GT 
			uniqueLabels_GT = self.unique2(np_mask_CC_GT)
			uniqueLabels_GT = np.sort(uniqueLabels_GT)
			## skip the first element, background is 0 
			uniqueLabels_GT = uniqueLabels_GT[1:]
			print('uniqueLabels_GT:', len(uniqueLabels_GT))
			print(uniqueLabels_GT)

			print('filtering small predictions that may be incorrect')

			## filter small predictions
			for GT_label in uniqueLabels_GT:
				np_mask_GT_label = np_mask_CC_GT == GT_label
				if np.sum(np_mask_GT_label) <= 10:
					itk_GT_connectedComponents = self.set_mask_value(itk_GT_connectedComponents, itk_GT_connectedComponents == GT_label, 0)

			## redo the CCA 
			itk_GT_connectedComponents = sitk.ConnectedComponent(itk_GT_connectedComponents > 0)
			
			## numpy GT 
			np_mask_CC_GT = sitk.GetArrayFromImage(itk_GT_connectedComponents)

			print('np_mask_CC_GT.shape:', np_mask_CC_GT.shape)

			## find unique labels in GT 
			uniqueLabels_GT = self.unique2(np_mask_CC_GT)
			uniqueLabels_GT = np.sort(uniqueLabels_GT)
			## skip the first element, background is 0 
			uniqueLabels_GT = uniqueLabels_GT[1:]
			print('uniqueLabels_GT:', len(uniqueLabels_GT))
			print(uniqueLabels_GT)

			num_ROI_in_all_GT += len(uniqueLabels_GT)
			

			"""
			#########################################################################################################################
			GT -- Stratify objects by size 
			For FASTER processing, precompute GT ROI above/below Xmm size - this is GT labels above/below Xmm
			Look at stratify function below for more details (uncomment code + work with it to get it what you want)
			"""	
			
			print('Stratifying GT by size:', len(uniqueLabels_GT))

			## stratify GT objects by size
			l_GT_labels_lesserThan_Xmm, l_GT_labels_greaterThan_Xmm = self.stratify_objects_by_size_GT(np_mask_CC_GT, uniqueLabels_GT, spacing_mask_GT)

			print('less than Xmm:', len(l_GT_labels_lesserThan_Xmm))
			print(l_GT_labels_lesserThan_Xmm)
			print('greater than Xmm:', len(l_GT_labels_greaterThan_Xmm))
			print(l_GT_labels_greaterThan_Xmm)
			
			## make GT images for different ROI sizes 
			np_mask_all_GT_labels_lesserThan_Xmm = np.zeros(shape = np_mask_CC_GT.shape, dtype = 'int')
			np_mask_all_GT_labels_greaterThan_Xmm = np.zeros(shape = np_mask_CC_GT.shape, dtype = 'int')

			for labelIdx in l_GT_labels_lesserThan_Xmm:
				np_mask_all_GT_labels_lesserThan_Xmm[np_mask_CC_GT == labelIdx] = labelIdx

			for labelIdx in l_GT_labels_greaterThan_Xmm:
				np_mask_all_GT_labels_greaterThan_Xmm[np_mask_CC_GT == labelIdx] = labelIdx
			

			"""
			#########################################################################################################################
			Predictions - Stratify objects by size
			Differentiate small from large objects based on length of minor Feret diameter
			90* normal Feret diameter is perpendicular to the max Feret Diameter 
			"""

			print('Stratifying Prediction by size:', len(uniqueLabels_pred))

			## stratify GT objects by size
			l_pred_labels_lesserThan_Xmm, l_pred_labels_greaterThan_Xmm = self.stratify_objects_by_size_prediction(np_mask_CC_pred, uniqueLabels_pred, spacing_mask_pred)

			print('pred_labels less than Xmm:', len(l_pred_labels_lesserThan_Xmm))
			print(l_pred_labels_lesserThan_Xmm)
			print('pred_labels greater than Xmm:', len(l_pred_labels_greaterThan_Xmm))
			print(l_pred_labels_greaterThan_Xmm)


			"""
			#########################################################################################################################
			Detection Performance
			"""		

			## comparing pred > Xmm against GT 	

			print('Computing detection metrics:')

			print('num pred_labels:', len(uniqueLabels_pred))

			def detect(					
						np_GT_CC,
						np_pred_CC,
						l_pred_labels,
					):
				
				l_found_labels_GT = []
				fp_ = 0

				for pred_label in l_pred_labels:
					## overlap of pred > Xmm with some GT 
					np_GTPred_overlap = np_GT_CC * (np_pred_CC == pred_label)
					## find unique overlapping GT
					t_uniqueLabels_GT = self.unique2(np_GTPred_overlap)
					t_uniqueLabels_GT = np.sort(t_uniqueLabels_GT)
					## exclude background (0)
					t_uniqueLabels_GT = t_uniqueLabels_GT[1:]
					## overlap exists? -- meaning [X, Y, Z], no 0 as background
					if len(t_uniqueLabels_GT) > 0:
						## store GT labels that overlap
						for x in t_uniqueLabels_GT:
							if x not in l_found_labels_GT:
								l_found_labels_GT.append(x)
					## no overlap, this is FP
					else:
						fp_ += 1

				## list of predicted ROI IDs that were found
				l_found_labels_GT = sorted(l_found_labels_GT)
				
				return l_found_labels_GT, fp_

						
			print('#########################')
			print('Computing detection metrics for ALL Objects')

			## find all TP GT label IDs that intersect with pred label IDs  
			l_GT_TP_all, pred_fp_all = detect(
													np_mask_CC_GT,
													np_mask_CC_pred,
													uniqueLabels_pred,
												)
			
			## calculate true positives and false negatives
			pred_tp_all = len(l_GT_TP_all)
			pred_fn_all = len(uniqueLabels_GT) - pred_tp_all

			## store  
			d_detectionMetrics['l_GT_all'].append(uniqueLabels_GT)
			d_detectionMetrics['l_numGT_all'].append(len(uniqueLabels_GT))
			d_detectionMetrics['l_GT_TP_all'].append(l_GT_TP_all)
			d_detectionMetrics['l_tp_all'].append(pred_tp_all)
			d_detectionMetrics['l_fp_all'].append(pred_fp_all)
			d_detectionMetrics['l_fn_all'].append(pred_fn_all)

			print('GT_all:', len(uniqueLabels_GT))
			print('pred_tp_all:', pred_tp_all)
			print('pred_fp_all:', pred_fp_all)
			print('pred_fn_all:', pred_fn_all)

			print('#########################')
			print('Computing detection metrics for Objects < Xmm')

			## find all TP GT label IDs that intersect with pred label IDs  
			l_GT_TP_lesserThan_Xmm, pred_fp_lesserThan_Xmm = detect(
																				np_mask_all_GT_labels_lesserThan_Xmm,
																				np_mask_CC_pred,
																				l_pred_labels_lesserThan_Xmm,
																			)
			
			## calculate true positives and false negatives
			pred_tp_lesserThan_Xmm = len(l_GT_TP_lesserThan_Xmm)
			pred_fn_lesserThan_Xmm = len(l_GT_labels_lesserThan_Xmm) - pred_tp_lesserThan_Xmm

			## store  
			d_detectionMetrics['l_GT_lesserThan_Xmm'].append(l_GT_labels_lesserThan_Xmm)
			d_detectionMetrics['l_numGT_lesserThan_Xmm'].append(len(l_GT_labels_lesserThan_Xmm))
			d_detectionMetrics['l_GT_TP_lesserThan_Xmm'].append(l_GT_TP_lesserThan_Xmm)
			d_detectionMetrics['l_tp_lesserThan_Xmm'].append(pred_tp_lesserThan_Xmm)
			d_detectionMetrics['l_fp_lesserThan_Xmm'].append(pred_fp_lesserThan_Xmm)
			d_detectionMetrics['l_fn_lesserThan_Xmm'].append(pred_fn_lesserThan_Xmm)

			print('numGT_lesserThan_Xmm:', len(l_GT_labels_lesserThan_Xmm))
			print('pred_tp_lesserThan_Xmm:', pred_tp_lesserThan_Xmm)
			print('pred_fp_lesserThan_Xmm:', pred_fp_lesserThan_Xmm)
			print('pred_fn_lesserThan_Xmm:', pred_fn_lesserThan_Xmm)

			print('#########################')
			print('Computing detection metrics for Objects >= Xmm')

			## find all TP GT label IDs that intersect with pred label IDs  
			l_GT_TP_greaterThan_Xmm, pred_fp_greaterThan_Xmm = detect(
																				np_mask_all_GT_labels_greaterThan_Xmm,
																				np_mask_CC_pred,
																				l_pred_labels_greaterThan_Xmm,
																			)
			
			## calculate true positives and false negatives
			pred_tp_greaterThan_Xmm = len(l_GT_TP_greaterThan_Xmm)
			pred_fn_greaterThan_Xmm = len(l_GT_labels_greaterThan_Xmm) - pred_tp_greaterThan_Xmm

			## store  
			d_detectionMetrics['l_GT_greaterThan_Xmm'].append(l_GT_labels_greaterThan_Xmm)
			d_detectionMetrics['l_numGT_greaterThan_Xmm'].append(len(l_GT_labels_greaterThan_Xmm))
			d_detectionMetrics['l_GT_TP_greaterThan_Xmm'].append(l_GT_TP_greaterThan_Xmm)
			d_detectionMetrics['l_tp_greaterThan_Xmm'].append(pred_tp_greaterThan_Xmm)
			d_detectionMetrics['l_fp_greaterThan_Xmm'].append(pred_fp_greaterThan_Xmm)
			d_detectionMetrics['l_fn_greaterThan_Xmm'].append(pred_fn_greaterThan_Xmm)
					  
			print('numGT_greaterThan_Xmm:', len(l_GT_labels_greaterThan_Xmm))
			print('pred_tp_greaterThan_Xmm:', pred_tp_greaterThan_Xmm)
			print('pred_fp_greaterThan_Xmm:', pred_fp_greaterThan_Xmm)
			print('pred_fn_greaterThan_Xmm:', pred_fn_greaterThan_Xmm)


		# 	# ########################################################################################################################
		# 	# ########################################################################################################################
		# 	## comparing GT against pred 

		# 	# print('Computing detection metrics:')

		# 	# # print('num pred_labels greaterThan_Xcm:', len(l_pred_labels_greaterThan_Xcm))
		# 	# l_found_labels_pred = []
		# 	# for labelIdx in l_GT_labels_greaterThan_Xcm:
		# 	# 	np_GTPred_overlap = (np_mask_GT == labelIdx) * np_mask_CC_pred
		# 	# 	## TP - overlap of current GT with some predicted LN
		# 	# 	if np.sum(np_GTPred_overlap) > 0: 
		# 	# 		t_uniqueLabels_pred = self.unique2(np_GTPred_overlap)
		# 	# 		t_uniqueLabels_pred = np.sort(t_uniqueLabels_pred)
		# 	# 		t_uniqueLabels_pred = t_uniqueLabels_pred[1:]
		# 	# 		for x in t_uniqueLabels_pred:
		# 	# 			if x not in l_found_labels_pred:
		# 	# 				l_found_labels_pred.append(x)
		# 	# 		curr_vol_tp += 1
		# 	# 	## FN - missed the current GT
		# 	# 	else:
		# 	# 		curr_vol_fn += 1
			
		# 	# ## list of predicted LN idxs found
		# 	# l_found_labels_pred = sorted(l_found_labels_pred)

		# 	# print('pred_labels greater than Xmm:', len(l_pred_labels_greaterThan_Xcm))
		# 	# print(l_pred_labels_greaterThan_Xcm)
		# 	# print('pred_labels found > Xmm:', len(l_found_labels_pred))
		# 	# print(l_found_labels_pred)

		# 	# if len(l_pred_labels_greaterThan_Xcm) > 1:
		# 	# 	## FP -- this is any predicted LN > Xmm that we did not capture 
		# 	# 	curr_vol_fp = len(set(l_pred_labels_greaterThan_Xcm) ^ set(l_found_labels_pred))
		# 	# else:
		# 	# 	curr_vol_fp = 0

		# 	# print('curr_vol GT:', len(uniqueLabels_GT))
		# 	# print('curr_vol GT > Xmm:', len(l_GT_labels_greaterThan_Xcm))
		# 	# print('curr_vol TP:', curr_vol_tp)
		# 	# print('curr_vol FP:', curr_vol_fp)
		# 	# print('curr_vol FN:', curr_vol_fn)

		# 	# ## add to counter
		# 	# total_GT += len(uniqueLabels_GT)
		# 	# total_GT_greaterThan_Xmm += len(l_GT_labels_greaterThan_Xcm)
		# 	# total_tp += curr_vol_tp
		# 	# total_fp += curr_vol_fp
		# 	# total_fn += curr_vol_fn

			"""
			#########################################################################################################################
			Segmentation Performance
			"""

			print('Computing segmentation metrics:')

			# v_test_dice_ALL = self.dice_coefficient(np_actual_mask_pred>0, np_mask_GT>0)
			# print('v_test_dice_ALL:', v_test_dice_ALL)
			# l_test_dice.append(v_test_dice_ALL)
			

			## make spacing for all eval
			spacing_mm = (spacing_mask_GT[2], spacing_mask_GT[0], spacing_mask_GT[1])

			print('#########################')
			print('Computing segmentation metrics for ALL Objects - Get overlap of GT with prediction CC')
			
			## all LN
			DSC_all, HD_all, simple_DSC_all, simple_HD_all = self.test_dice_HD(
																					np_mask_CC_GT, 
																					np_mask_CC_pred, 
																					spacing_mm
																				)
			
			print("Dice: {}, HD_95: {} mm".format(DSC_all, HD_all))
			print("Simple -- Dice: {}, HD_95: {} mm".format(simple_DSC_all, simple_HD_all))

			## store dice 
			d_DSC['l_DSC_all'].append(DSC_all)
			d_DSC['l_DSC_all_simple'].append(simple_DSC_all)

			## store hausdorff 
			d_HD95['l_HD_all'].append(HD_all)
			d_HD95['l_HD_all_simple'].append(simple_HD_all)

			print('#########################')
			print('Computing segmentation metrics for Objects < Xmm - Get overlap of GT with prediction CC')
			
			## lesser than Xmm
			DSC_lesserThan_Xmm, HD_lesserThan_Xmm, simple_DSC_LNbelowXmm, simple_HD_LNbelowXmm = self.test_dice_HD(
																													np_mask_all_GT_labels_lesserThan_Xmm, 
																													np_mask_CC_pred, 
																													spacing_mm
																												)
			
			print("Dice: {}, HD_95: {} mm".format(DSC_lesserThan_Xmm, HD_lesserThan_Xmm))
			print("Simple -- Dice: {}, HD_95: {} mm".format(simple_DSC_LNbelowXmm, simple_HD_LNbelowXmm))

			## store dice 
			d_DSC['l_DSC_lesserThan_Xmm'].append(DSC_lesserThan_Xmm)
			d_DSC['l_DSC_lesserThan_Xmm_simple'].append(simple_DSC_LNbelowXmm)			

			## store hausdorff 
			d_HD95['l_HD_lesserThan_Xmm'].append(HD_lesserThan_Xmm)
			d_HD95['l_HD_lesserThan_Xmm_simple'].append(simple_HD_LNbelowXmm)
			

			print('#########################')
			print('Computing segmentation metrics for Objects >= Xmm - Get overlap of GT with prediction CC')

			## greater than Xmm
			DSC_greaterThan_Xmm, HD_greaterThan_Xmm, simple_DSC_LNoverXmm, simple_HD_LNoverXmm = self.test_dice_HD(
																													np_mask_all_GT_labels_greaterThan_Xmm, 
																													np_mask_CC_pred, 
																													spacing_mm
																												)
		
			print("Dice: {}, HD_95: {} mm".format(DSC_greaterThan_Xmm, HD_greaterThan_Xmm))
			print("Simple -- Dice: {}, HD_95: {} mm".format(simple_DSC_LNoverXmm, simple_HD_LNoverXmm))

			## store dice
			d_DSC['l_DSC_greaterThan_Xmm'].append(DSC_greaterThan_Xmm)
			d_DSC['l_DSC_greaterThan_Xmm_simple'].append(simple_DSC_LNoverXmm)

			## store hausdorff 
			d_HD95['l_HD_greaterThan_Xmm'].append(HD_greaterThan_Xmm)
			d_HD95['l_HD_greaterThan_Xmm_simple'].append(simple_HD_LNoverXmm)


			"""
			#########################################################################################################################
			Volume Computation
			"""

			## get volume for 1 voxel
			unit_volume = np.prod(itk_mask_GT.GetSpacing())			

			## volume for GT
			true_volume_voxels = np.sum(np_mask_CC_GT > 0) 
			true_volume = true_volume_voxels * unit_volume * 1e-3  # for solids - cm3 or cc or cubic centimeter; for fluids - mL 
			
			## volume for pred
			pred_volume_voxels = np.sum(np_mask_CC_pred > 0)
			pred_volume = pred_volume_voxels * unit_volume * 1e-3  # for solids - cm3 or cc or cubic centimeter; for fluids - mL
			
			## store
			d_volumeCalc['l_vol_true_voxels'].append(true_volume_voxels)
			d_volumeCalc['l_vol_true_cc'].append(true_volume)

			## store
			d_volumeCalc['l_vol_pred_voxels'].append(pred_volume_voxels)
			d_volumeCalc['l_vol_pred_cc'].append(pred_volume)

			## store
			d_volumeCalc['l_unit_volume'].append(unit_volume)


		print()
		print('================================')		
		print('Final Segmentation performance')
		print('================================')
		print() 

		np.set_printoptions(precision=3)

		#print('test_dice_ALL:', np.mean(l_test_dice), np.std(l_test_dice))

		print('--'*5, 'Computing metrics for ALL Objects', '--'*5)
		
		l_vars = [v for v in d_DSC['l_DSC_all'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('dice:', np.mean(l_vars), np.std(l_vars))

		l_vars = [v for v in d_DSC['l_DSC_all_simple'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('simple_dice:', np.mean(l_vars), np.std(l_vars))
		
		l_vars = [v for v in d_HD95['l_HD_all'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('HD:', np.mean(l_vars), np.std(l_vars))		
		
		l_vars = [v for v in d_HD95['l_HD_all_simple'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('simple_HD:', np.mean(l_vars), np.std(l_vars))

		print() 
		print('--'*5, 'Computing metrics for Objects < Xmm', '--'*5)

		l_vars = [v for v in d_DSC['l_DSC_lesserThan_Xmm'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('dice:', np.mean(l_vars), np.std(l_vars))

		l_vars = [v for v in d_DSC['l_DSC_lesserThan_Xmm_simple'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('simple_dice:', np.mean(l_vars), np.std(l_vars))
		
		l_vars = [v for v in d_HD95['l_HD_lesserThan_Xmm'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('HD:', np.mean(l_vars), np.std(l_vars))
		
		l_vars = [v for v in d_HD95['l_HD_lesserThan_Xmm_simple'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('simple_HD:', np.mean(l_vars), np.std(l_vars))

		print() 
		print('--'*5, 'Computing metrics for Objects >= Xmm', '--'*5)

		l_vars = [v for v in d_DSC['l_DSC_greaterThan_Xmm'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('dice:', np.mean(l_vars), np.std(l_vars))

		l_vars = [v for v in d_DSC['l_DSC_greaterThan_Xmm_simple'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('simple_dice:', np.mean(l_vars), np.std(l_vars))

		l_vars = [v for v in d_HD95['l_HD_greaterThan_Xmm'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('HD:', np.mean(l_vars), np.std(l_vars))

		l_vars = [v for v in d_HD95['l_HD_greaterThan_Xmm_simple'] if not math.isnan(float(v)) and not math.isinf(float(v))]
		print('simple_HD:', np.mean(l_vars), np.std(l_vars))

		print()
		print('================================')		
		print('Final Detection performance')
		print('================================')
		print() 

		print('--'*5, 'Computing metrics for ALL Objects', '--'*5)

		total_GT_all = np.sum(d_detectionMetrics['l_numGT_all'])
		total_tp_all = np.sum(d_detectionMetrics['l_tp_all'])
		total_fp_all = np.sum(d_detectionMetrics['l_fp_all'])
		total_fn_all = np.sum(d_detectionMetrics['l_fn_all'])

		precision_all = total_tp_all / (total_tp_all + total_fp_all)
		recall_all = total_tp_all / (total_tp_all + total_fn_all)
		F1_all = (2*total_tp_all) / ((2*total_tp_all) + total_fp_all + total_fn_all)

		print('total_GT_all:', total_GT_all)
		print('total_tp_all:', total_tp_all)
		print('total_fp_all:', total_fp_all)
		print('total_fn_all:', total_fn_all)
		print('precision_all:', precision_all)
		print('recall_all:', recall_all)
		print('F1_all:', F1_all)
		
		## store detection metrics
		d_detectionMetrics['total_tp_all'] = total_tp_all
		d_detectionMetrics['total_fp_all'] = total_fp_all
		d_detectionMetrics['total_fn_all'] = total_fn_all
		d_detectionMetrics['precision_all'] = precision_all
		d_detectionMetrics['recall_all'] = recall_all
		d_detectionMetrics['F1_all'] = F1_all

		print() 
		print('--'*5, 'Computing metrics for Objects < Xmm', '--'*5)

		total_GT_lesserThan_Xmm = np.sum(d_detectionMetrics['l_numGT_lesserThan_Xmm'])
		total_tp_lesserThan_Xmm = np.sum(d_detectionMetrics['l_tp_lesserThan_Xmm'])
		total_fp_lesserThan_Xmm = np.sum(d_detectionMetrics['l_fp_lesserThan_Xmm'])
		total_fn_lesserThan_Xmm = np.sum(d_detectionMetrics['l_fn_lesserThan_Xmm'])

		precision_lesserThan_Xmm = total_tp_lesserThan_Xmm / (total_tp_lesserThan_Xmm + total_fp_lesserThan_Xmm)
		recall_lesserThan_Xmm = total_tp_lesserThan_Xmm / (total_tp_lesserThan_Xmm + total_fn_lesserThan_Xmm)
		F1_lesserThan_Xmm = (2*total_tp_lesserThan_Xmm) / ((2*total_tp_lesserThan_Xmm) + total_fp_lesserThan_Xmm + total_fn_lesserThan_Xmm)

		print('total_GT_lesserThan_Xmm:', total_GT_lesserThan_Xmm)
		print('total_tp_lesserThan_Xmm:', total_tp_lesserThan_Xmm)
		print('total_fp_lesserThan_Xmm:', total_fp_lesserThan_Xmm)
		print('total_fn_lesserThan_Xmm:', total_fn_lesserThan_Xmm)
		print('precision_lesserThan_Xmm:', precision_lesserThan_Xmm)
		print('recall_lesserThan_Xmm:', recall_lesserThan_Xmm)
		print('F1_lesserThan_Xmm:', F1_lesserThan_Xmm)
		
		## store detection metrics
		d_detectionMetrics['total_tp_lesserThan_Xmm'] = total_tp_lesserThan_Xmm
		d_detectionMetrics['total_fp_lesserThan_Xmm'] = total_fp_lesserThan_Xmm
		d_detectionMetrics['total_fn_lesserThan_Xmm'] = total_fn_lesserThan_Xmm
		d_detectionMetrics['precision_lesserThan_Xmm'] = precision_lesserThan_Xmm
		d_detectionMetrics['recall_lesserThan_Xmm'] = recall_lesserThan_Xmm
		d_detectionMetrics['F1_lesserThan_Xmm'] = F1_lesserThan_Xmm

		print() 
		print('--'*5, 'Computing metrics for Objects >= Xmm', '--'*5)

		total_GT_greaterThan_Xmm = np.sum(d_detectionMetrics['l_numGT_greaterThan_Xmm'])
		total_tp_greaterThan_Xmm = np.sum(d_detectionMetrics['l_tp_greaterThan_Xmm'])
		total_fp_greaterThan_Xmm = np.sum(d_detectionMetrics['l_fp_greaterThan_Xmm'])
		total_fn_greaterThan_Xmm = np.sum(d_detectionMetrics['l_fn_greaterThan_Xmm'])

		precision_greaterThan_Xmm = total_tp_greaterThan_Xmm / (total_tp_greaterThan_Xmm + total_fp_greaterThan_Xmm)
		recall_greaterThan_Xmm = total_tp_greaterThan_Xmm / (total_tp_greaterThan_Xmm + total_fn_greaterThan_Xmm)
		F1_greaterThan_Xmm = (2*total_tp_greaterThan_Xmm) / ((2*total_tp_greaterThan_Xmm) + total_fp_greaterThan_Xmm + total_fn_greaterThan_Xmm)

		print('total_GT_greaterThan_Xmm:', total_GT_greaterThan_Xmm)
		print('total_tp_greaterThan_Xmm:', total_tp_greaterThan_Xmm)
		print('total_fp_greaterThan_Xmm:', total_fp_greaterThan_Xmm)
		print('total_fn_greaterThan_Xmm:', total_fn_greaterThan_Xmm)
		print('precision_greaterThan_Xmm:', precision_greaterThan_Xmm)
		print('recall_greaterThan_Xmm:', recall_greaterThan_Xmm)
		print('F1_greaterThan_Xmm:', F1_greaterThan_Xmm)
		
		## store detection metrics
		d_detectionMetrics['total_tp_greaterThan_Xmm'] = total_tp_greaterThan_Xmm
		d_detectionMetrics['total_fp_greaterThan_Xmm'] = total_fp_greaterThan_Xmm
		d_detectionMetrics['total_fn_greaterThan_Xmm'] = total_fn_greaterThan_Xmm
		d_detectionMetrics['precision_greaterThan_Xmm'] = precision_greaterThan_Xmm
		d_detectionMetrics['recall_greaterThan_Xmm'] = recall_greaterThan_Xmm
		d_detectionMetrics['F1_greaterThan_Xmm'] = F1_greaterThan_Xmm

		
		return l_cases, d_detectionMetrics, d_DSC, d_HD95, d_volumeCalc
		
		
	#########################################################################################################################

	def run_evaluation(self, str_fn_2_save):
		
		## get all mask files in folder
		self.l_ffpn_masks_GT = self.get_list_filesInFolder(self.dir_mask_GT)
		self.l_ffpn_masks_pred = self.get_list_filesInFolder(self.dir_mask_pred)

		# print('l_ffpn_masks_pred')
		# print(self.l_ffpn_masks_pred)

		## assert same number of GT and preds
		assert len(self.l_ffpn_masks_GT) == len(self.l_ffpn_masks_pred)

		print('Begin metric computation:')

		## compute metrics
		l_cases, d_detectionMetrics, d_DSC, d_HD95, d_volumeCalc = self.compute_metrics_simpleITK(self.l_ffpn_masks_GT, self.l_ffpn_masks_pred)

		print('collating results:')

		## collate metric results into dataframe
		df_results = self.collate_results_(l_cases, d_DSC, d_HD95, d_volumeCalc)

		print('dumping results to disk')

		## dump metric results to csv
		self.dump_results_to_csv(df_results, str_fn_2_save,)


#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
		

if __name__ == "__main__":

	# on pipnas
	dir_root = '/mnt/cc2pipnas/groups/SummersLab/datasets/public/unitochest/nii/' 
	# ## on supermicro2
	# dir_root = '/storage2/tsm/datasets_nnUNet/nnUNet_raw/' ## on supermicro2

	# on pipnas
	dir_mask_GT = dir_root + 'labelsTs_original/' 
	# ## on supermicro2
	# dir_mask_GT = dir_root + '/' 
	
	## low-res 
	# dir_mask_pred = dir_root + '/' 
	# # full-res 
	# dir_mask_pred = dir_root + 'predictLabels_unitochest_169vols_29Structures_withAnatomyPriors/'  
	# full-res 
	dir_mask_pred = dir_root + 'predictLabels_unitochest_169vols_29Structures_noAnatomyPriors_ablationExpOnlyNodules/'  
	## cascade unet (with low-res)
	# dir_mask_pred = dir_root + '/' 
	# ## cascade unet (with full-res)
	# dir_mask_pred = dir_root + '/' 
	
	## ablation experiment - only LN with no anatomical priors
	# dir_mask_pred = dir_root + '/' 
	
	# str_fn_2_save = 'unitochest_results_nnUNetLowRes'
	# str_fn_2_save = 'unitochest_results_nnUNetFullRes'
	# str_fn_2_save = 'unitochest_results_nnUNetCascadeLowRes'
	# str_fn_2_save = 'unitochest_results_nnUNetCascadeFullRes'
	
	# ## nnUNet 3D full-res (with 28 TS anatomy priors)
	# str_fn_2_save = 'unitochest_results_nnUNetFullRes_withAnatomyPriors'
	## ablation
	str_fn_2_save = 'unitochest_results_nnUNetFullRes_noAnatomyPriors_ablationExpOnlyNodules'

	print('dir_mask_GT:')
	print(dir_mask_GT)
	print('dir_mask_pred:')
	print(dir_mask_pred)

	## create class instance
	lncte = LNCT_evaluation(dir_mask_GT, dir_mask_pred)

	## run
	lncte.run_evaluation(str_fn_2_save)





