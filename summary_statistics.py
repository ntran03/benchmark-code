from scipy import stats
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import os


tools = ["ts", "mr", "vibe", "mri"]
dice = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23] #[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
hd = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24] #[3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
class_list =  ["arterial", "delayed", "precontrast", "venous"]

def combine_data():
    '''creates one giant csv with all the volumes and an extra column for the tool used
    '''
    for tool in tools:
        temp_df = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/{tool}/arterial.csv")
        temp_df.insert(1, "Class", "arterial")
        #append all the sequence types together for one tool
        for sequence in class_list[1:]:
            seq = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/{tool}/{sequence}.csv")
            seq.insert(1, "Class", sequence)
            temp_df = temp_df._append(seq, ignore_index=True)
        if tool == "ts":
            temp_df.insert(0, "Method", "TS")
            final_df = temp_df
        else:
            if (tool == "mr") or (tool == "mri"):
                tool_word = f"{tool.upper()}Seg"
            else:
                tool_word = tool.upper()
            temp_df.insert(0, "Method", tool_word)
            final_df = pd.concat([final_df, temp_df], ignore_index=True)
    final_df.to_csv("/Users/nicol/Documents/nih/outputs/all_data.csv")
                
def clean_data():
    '''gets rid of the inf hd values
    '''
    
    uncleaned = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/all_data.csv")

    # Step 1: Identify instances of inf and record values in the 4th column
    inf_rows = uncleaned[uncleaned.isin([np.inf]).any(axis=1)]
    recorded_values = inf_rows.iloc[:, 4].unique()

    # Step 2: Iterate through each row that has an inf value
    for index, row in inf_rows.iterrows():
        # Record the value in the third column
        scan_id = row.iloc[4]
        
        # Find columns with inf value
        inf_columns = row[row == np.inf].index

        # Delete the inf instance (hd only)
        for col in inf_columns:
            col_index = uncleaned.columns.get_loc(col)
            uncleaned.at[index, col] = np.nan
            
        
        # Step 3: For rows with the same value in the 4th column, delete the same two cells
        same_value_rows = uncleaned[uncleaned.iloc[:, 4] == scan_id].index
        for idx in same_value_rows:
            for col in inf_columns:
                col_index = uncleaned.columns.get_loc(col)
                uncleaned.at[idx, col] = np.nan
        
    # Step 4: Save the modified DataFrame back to a CSV file
    uncleaned.to_csv(f"/Users/nicol/Documents/nih/outputs/modified_file.csv", index=False)
    
def get_clean_data(tool = "all", sequence = "all", organ=None):
    '''grabs data from cleaned df
    '''
    temp = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/modified_file.csv")
    
    # Filter rows by the specified tool and sequence
    if sequence != "all":
        filtered_df = temp[(temp.iloc[:, 1] == tool) & (temp.iloc[:, 3] == sequence)]
    else:
        if organ != None:
            hd_temp = temp.loc[temp["Method"] == tool, f"hd95_{organ}"]
            dsc_temp =  temp.loc[temp["Method"] == tool, f"dsc_{organ}"]
            hd_series = pd.Series(hd_temp)
            # Ensure all values are numeric, convert non-numeric to NaN
            hd_series = hd_series.apply(pd.to_numeric, errors='coerce')
            # Drop rows with NaN values 
            hd_series = hd_series.dropna()
            dice_series = pd.Series(dsc_temp)
            return dice_series, hd_series
        else:
            filtered_df = temp[(temp.iloc[:, 1] == tool)]
        
        
    dice_df = filtered_df.iloc[:, dice] #get all dsc cols
    hd_df = filtered_df.iloc[:, hd] #get all hd cols
    
    dice_series = dice_df.values.flatten()
    hd_series = hd_df.values.flatten()
    hd_series = pd.Series(hd_series)
    # Ensure all values are numeric, convert non-numeric to NaN
    hd_series = hd_series.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values 
    hd_series = hd_series.dropna()
    
    dice_series = pd.Series(dice_series)
    
    return dice_series, hd_series

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
    '''does friedman on one sequence type for ts, mrseg, vibe, mriseg
    '''
    print(sequence)
    
    #grab data
    tseg_dsc, tseg_hd = get_clean_data("TS", sequence)
    mrseg_dsc, mrseg_hd = get_clean_data("MRSeg", sequence)
    vibe_dsc, vibe_hd = get_clean_data("VIBE", sequence)
    mriseg_dsc, mriseg_hd = get_clean_data("MRISeg", sequence)
    
    print("dsc")
    
    #perform Friedman Test
    print(stats.friedmanchisquare(tseg_dsc, mrseg_dsc, vibe_dsc, mriseg_dsc))

    # Combine four groups into one array
    data_dsc = np.array([tseg_dsc, mrseg_dsc, vibe_dsc, mriseg_dsc])
    # # Convert the NumPy array to a pandas DataFrame
    # df = pd.DataFrame(data_dsc.T, columns=['ts', 'mr', 'vibe', "mri"])

    # # Save the DataFrame to a CSV file
    # df.to_csv(f"/Users/nicol/Documents/nih/outputs/testing.csv", index=False)

    
    # Conduct the Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data_dsc.T))
    
    print("hd")
    
    #perform Friedman Test
    print(stats.friedmanchisquare(tseg_hd, mrseg_hd, vibe_hd, mriseg_hd))

    # Combine four groups into one array
    data_hd = np.array([tseg_hd, mrseg_hd, vibe_hd, mriseg_hd])
    
    # Conduct the Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data_hd.T))
    
def friedman_organ():
    df = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/modified_file.csv")
    for organ in ["spleen", "right_kidney", "left_kidney", "stomach", "pancreas", "right_adrenal_gland", "left_adrenal_gland", "aorta", "inferior_vena_cava", "liver"]:
        print(organ)
        
        dsc_col = []
        hd_col = []
        for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
            dsc_temp, hd_temp = get_clean_data(tool=tool, sequence="all", organ=organ)
            dsc_col.append(dsc_temp)
            hd_col.append(hd_temp)
        
        print("dsc")
        
        print(stats.friedmanchisquare(dsc_col[0], dsc_col[1], dsc_col[2], dsc_col[3]))
        
        data_dsc = np.array([dsc_col[0], dsc_col[1], dsc_col[2], dsc_col[3]])
        
        print(sp.posthoc_nemenyi_friedman(data_dsc.T))
    
        print("hd")
        
        print(stats.friedmanchisquare(hd_col[0], hd_col[1], hd_col[2], hd_col[3]))
        
        data_dsc = np.array([hd_col[0], hd_col[1], hd_col[2], hd_col[3]])
        
        print(sp.posthoc_nemenyi_friedman(data_dsc.T)) 
        
def overall():
    tseg_dsc, tseg_hd = get_clean_data("TS", "all")
    mrseg_dsc, mrseg_hd = get_clean_data("MRSeg", "all")
    vibe_dsc, vibe_hd = get_clean_data("VIBE", "all")
    mriseg_dsc, mriseg_hd = get_clean_data("MRISeg", "all")
    
    filtered_dsc = [tseg_dsc, mrseg_dsc, vibe_dsc, mriseg_dsc]
    filtered_hd = [tseg_hd, mrseg_hd, vibe_hd, mriseg_hd]
    
    
    print("DSC")
    print(f"TS: {filtered_dsc[0].mean()}")
    print(f"MRSeg: {filtered_dsc[1].mean()}")
    print(f"VIBE: {filtered_dsc[2].mean()}")
    print(f"MRISeg: {filtered_dsc[3].mean()}")
    
    group1 = filtered_dsc[0]
    group2 = filtered_dsc[1]
    group3 = filtered_dsc[2]
    group4 = filtered_dsc[3]
    

    print(stats.friedmanchisquare(group1, group2, group3, group4))
    
    data = np.array([group1, group2, group3, group4])
 
    # Conduct the Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data.T))

    print("HD")
    print(f"TS: {filtered_hd[0].mean()}")
    print(f"MRSeg: {filtered_hd[1].mean()}")
    print(f"VIBE: {filtered_hd[2].mean()}")
    print(f"MRISeg: {filtered_hd[3].mean()}")
    
    group1 = filtered_hd[0]
    group2 = filtered_hd[1]
    group3 = filtered_hd[2]
    group4 = filtered_hd[3]

    print(stats.friedmanchisquare(group1, group2, group3, group4))
    
    data = np.array([group1, group2, group3, group4])
 
    # Conduct the Nemenyi post-hoc test
    print(sp.posthoc_nemenyi_friedman(data.T))

def calculate_stats(data):
    mean_val = data.mean()
    std_dev = data.std()
    conf_interval = 1.96 * (std_dev / np.sqrt(len(data)))  # 95% CI assuming normal distribution
    return mean_val, std_dev, conf_interval

def all_means():  
    dsc_stats = {}
    hd_stats = {}
    for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
        dsc_temp, hd_temp = get_clean_data(tool=tool, sequence="all")
        
        dsc_mean, dsc_std, dsc_ci = calculate_stats(dsc_temp)
        hd_mean, hd_std, hd_ci = calculate_stats(hd_temp)

        dsc_stats[tool] = (dsc_mean, dsc_std, dsc_ci)
        hd_stats[tool] = (hd_mean, hd_std, hd_ci)
            
    print("DSC Metrics:")
    for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
        print(f"{tool}: Mean={dsc_stats[tool][0]:.3f}, SD={dsc_stats[tool][1]:.3f}, 95% CI={dsc_stats[tool][2]:.3f}")

    print("HD Metrics:")
    for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
        print(f"{tool}: Mean={hd_stats[tool][0]:.3f}, SD={hd_stats[tool][1]:.3f}, 95% CI={hd_stats[tool][2]:.3f}")


def organ_means():
    for organ in ["spleen", "right_kidney", "left_kidney", "stomach", "pancreas", "right_adrenal_gland", "left_adrenal_gland", "aorta", "inferior_vena_cava", "liver"]:
        print(organ)
        dsc_stats = {}
        hd_stats = {}
        for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
            dsc_temp, hd_temp = get_clean_data(tool=tool, sequence="all", organ=organ)
            
            dsc_mean, dsc_std, dsc_ci = calculate_stats(dsc_temp)
            hd_mean, hd_std, hd_ci = calculate_stats(hd_temp)

            dsc_stats[tool] = (dsc_mean, dsc_std, dsc_ci)
            hd_stats[tool] = (hd_mean, hd_std, hd_ci)
            
        print("DSC Metrics:")
        for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
            print(f"{tool}: Mean={dsc_stats[tool][0]:.3f}, SD={dsc_stats[tool][1]:.3f}, 95% CI={dsc_stats[tool][2]:.3f}")

        print("HD Metrics:")
        for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
            print(f"{tool}: Mean={hd_stats[tool][0]:.3f}, SD={hd_stats[tool][1]:.3f}, 95% CI={hd_stats[tool][2]:.3f}")

    
def sequence_means():
    for sequence in class_list:
        print(sequence)
        dsc_stats = {}
        hd_stats = {}
        
        for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
            dsc_temp, hd_temp = get_clean_data(tool=tool, sequence=sequence)
            
            dsc_mean, dsc_std, dsc_ci = calculate_stats(dsc_temp)
            hd_mean, hd_std, hd_ci = calculate_stats(hd_temp)

            dsc_stats[tool] = (dsc_mean, dsc_std, dsc_ci)
            hd_stats[tool] = (hd_mean, hd_std, hd_ci)
            
        print("DSC Metrics:")
        for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
            print(f"{tool}: Mean={dsc_stats[tool][0]:.3f}, SD={dsc_stats[tool][1]:.3f}, 95% CI={dsc_stats[tool][2]:.3f}")

        print("HD Metrics:")
        for tool in ["TS", "MRSeg", "VIBE", "MRISeg"]:
            print(f"{tool}: Mean={hd_stats[tool][0]:.3f}, SD={hd_stats[tool][1]:.3f}, 95% CI={hd_stats[tool][2]:.3f}")

    
#means/CI/std dev overall
all_means()

#means/CI/std dev sequencewise, 4 sequences 4 tools
#sequence_means()

#means/CI/std dev organwise, 4 sequences 10 organs 
#organ_means()

#sequence by sequence tool comparison
#for sequence in ["precontrast","arterial","delayed", "venous"]:
#    friedman(sequence)

#overall tool comparison
#overall()

#organ by organ tool comparison
#friedman_organ()

#combine_data()
#clean_data()