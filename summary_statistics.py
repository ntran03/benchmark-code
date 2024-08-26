from scipy import stats
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import os


tools = ["ts", "mr", "vibe"]
dice = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
hd = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

group1 = [4, 6, 3, 4, 3, 2, 2, 7, 6, 5]
group2 = [5, 6, 8, 7, 7, 8, 4, 6, 4, 5]
group3 = [2, 4, 4, 3, 2, 2, 1, 4, 3, 2]

def get_data(tool, sequence):
    
    temp = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/{tool}/{sequence}.csv")
    dice_df = temp.iloc[:, dice] #get all dsc cols
    hd_df = temp.iloc[:, hd] #get all hd cols
    
    dice_series = dice_df.values.flatten()
    hd_series = hd_df.values.flatten()
    
    # Identify indices where hd_series is not infinite
    valid_indices = ~np.isinf(hd_series)

    # Filter out the rows where hd_series is infinite
    filtered_dice_series = dice_series[valid_indices]
    filtered_hd_series = hd_series[valid_indices]
    
    return filtered_dice_series, filtered_hd_series


def friedman(sequence):
    
    print(sequence)
    
    #grab data
    tseg_dsc, tseg_hd = get_data("ts", sequence)
    mrseg_dsc, mrseg_hd = get_data("mr", sequence)
    vibe_dsc, vibe_hd = get_data("vibe", sequence)
    
    print("dsc")
    
    #perform Friedman Test
    print(stats.friedmanchisquare(tseg_dsc, mrseg_dsc, vibe_dsc))

    # Combine three groups into one array
    data_dsc = np.array([tseg_dsc, mrseg_dsc, vibe_dsc])
    
    # Conduct the Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data_dsc.T))
    
    print("hd")
    
    #perform Friedman Test
    print(stats.friedmanchisquare(tseg_hd, mrseg_hd, vibe_hd))

    # Combine three groups into one array
    data_hd = np.array([tseg_hd, mrseg_hd, vibe_hd])
    
    # Conduct the Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data_hd.T))
    
    
def overall():
    dice_list = []
    hd_list = []
    for tool in tools:
        csv_files = [file for file in os.listdir(f"/Users/nicol/Documents/nih/outputs/{tool}") if file.endswith(".csv")]
        for file in csv_files:
            temp = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/{tool}/{file}")
            dice_df = temp.iloc[:, dice]
            
            hd_df = temp.iloc[:, hd]
            dice_series = dice_df.values.flatten() #100 values, 10 per volume (organs) and 10 volumes per sequence type
            hd_series = hd_df.values.flatten()
            dice_list.append(dice_series)
            hd_list.append(hd_series)
    
    
    # Create a DataFrame where each column is one of the series
    dice_df = pd.DataFrame(dice_list).T
    hd_df = pd.DataFrame(hd_list).T


    # Group the DataFrame into chunks of 4 columns each, each chunk being a different tool
    dice_chunks = [dice_df.iloc[:, i:i+4] for i in range(0, dice_df.shape[1], 4)]
    hd_chunks = [hd_df.iloc[:, i:i+4] for i in range(0, hd_df.shape[1], 4)]

    # Combine the series in each chunk
    combined_dice = [chunk.stack().reset_index(drop=True) for chunk in dice_chunks]
    combined_hd = [chunk.stack().reset_index(drop=True) for chunk in hd_chunks]

    filtered_hd = []
    filtered_dsc = []
    print(len(combined_dice[0]))
    print(len(combined_hd[0]))
    
    # Clean combined_dice: remove inf values
    for i in range(len(combined_hd)):
        hd_series = combined_hd[i]
        dsc_series = combined_dice[i]
        valid_indices = ~np.isinf(hd_series)
        filtered_dice_series = dsc_series[valid_indices]
        filtered_hd_series = hd_series[valid_indices]
        filtered_dsc.append(filtered_dice_series)
        filtered_hd.append(filtered_hd_series)

    print("DSC")
    print(f"TS: {filtered_dsc[0].mean()}")
    print(f"MRSeg: {filtered_dsc[1].mean()}")
    print(f"VIBE: {filtered_dsc[2].mean()}")
    
    group1 = filtered_dsc[0]
    group2 = filtered_dsc[1]
    group3 = filtered_dsc[2]

    print(stats.friedmanchisquare(group1, group2, group3))
    
    data = np.array([group1, group2, group3])
 
    # Conduct the Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data.T))

    print("HD")
    print(f"TS: {filtered_hd[0].mean()}")
    print(f"MRSeg: {filtered_hd[1].mean()}")
    print(f"VIBE: {filtered_hd[2].mean()}")
    
    group1 = filtered_hd[0]
    group2 = filtered_hd[1]
    group3 = filtered_hd[2]


    print(stats.friedmanchisquare(group1, group2, group3))
    
    data = np.array([group1, group2, group3])
 
    # Conduct the Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data.T))


for sequence in ["precontrast","arterial","delayed", "venous"]:
    friedman(sequence)
#overall()