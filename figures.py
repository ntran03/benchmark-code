import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import ttest_rel


tools = ["ts", "mr", "vibe"]
dice = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
hd = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

plt.rcParams.update({
    'font.size': 20,          # Font size for all text elements
    'axes.titlesize': 22,     # Font size for axis titles
    'axes.labelsize': 20,     # Font size for axis labels
    'xtick.labelsize': 20,    # Font size for x-axis tick labels
    'ytick.labelsize': 20,    # Font size for y-axis tick labels
    'legend.fontsize': 20,    # Font size for legend
    'figure.titlesize': 25,   # Font size for figure title
})


def overall_boxplot():

    dice_series_list = []
    hd_series_list = []
    tool_labels = []


    for tool in tools:

        csv_files = os.listdir(f"/data/drdcad/nicole/outputs/comparison/{tool}")

        df_list = [pd.read_csv(f"/data/drdcad/nicole/outputs/comparison/{tool}/{file}") for file in csv_files]
        combined_df = pd.concat(df_list, ignore_index=True)

        # Extract values from the specified columns
        dice_df = combined_df.iloc[:, dice]
        
        # Create a DataFrame with ones in the specified columns
        hd_df = combined_df.iloc[:, hd]
        
        # Flatten the values_df and ones_df into Series
        dice_series = dice_df.values.flatten()
        hd_series = hd_df.values.flatten()

        # Extend the series lists with the current tool's data
        dice_series_list.extend(dice_series)
        hd_series_list.extend(hd_series)
        
        # Add tool labels to tool_labels list

        tool_labels.extend([tool] * len(dice_series))

        # Create DataFrames with a tool identifier
        dice_df_final = pd.DataFrame({
            'Value': dice_series_list,
            'Tool': tool_labels[:len(dice_series_list)]
        })

        hd_df_final = pd.DataFrame({
            'Value': hd_series_list,
            'Tool': tool_labels[:len(dice_series_list)]
        })

    # Define a color palette for the tools
    palette = sns.color_palette("Set2", n_colors=len(tools))

    # Plotting boxplots
    plt.figure(figsize=(15, 8))

    # Boxplot for dice series
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Tool', y='Value', data=dice_df_final, palette=palette)
    plt.title('Dice Series Comparison')
    plt.xticks(ticks=range(len(tools)), labels=["TSeg", "MRSeg", "Vibe"])

    # Boxplot for hd series
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Tool', y='Value', data=hd_df_final, palette=palette)
    plt.title('HD Series Comparison')
    plt.xticks(ticks=range(len(tools)), labels=["TSeg", "MRSeg", "Vibe"])


    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('comparison_boxplots.png')  # You can specify the file name and format (e.g., .png, .pdf)

    # Close the plot to free up memory
    plt.close()

def get_averages():
    for tool in tools:
        print(tool)
        csv_files = os.listdir(f"/data/drdcad/nicole/outputs/comparison/{tool}")
        for file in csv_files:
            temp = pd.read_csv(f"/data/drdcad/nicole/outputs/comparison/{tool}/{file}")
            dice_df = temp.iloc[:, dice]
            hd_df = temp.iloc[:, hd]
            dice_series = dice_df.values.flatten()
            hd_series = hd_df.values.flatten()
            print(f"Dice score for {file}: {dice_series.mean()}")
            print(f"HD for {file}: {hd_series.mean()}")

def get_organ_averages():
    means_list = []
    for tool in tools:
        print(tool)
        csv_files = os.listdir(f"/data/drdcad/nicole/outputs/comparison/{tool}")
        df_list = [pd.read_csv(f"/data/drdcad/nicole/outputs/comparison/{tool}/{file}") for file in csv_files]
        df = pd.concat(df_list, ignore_index=True)
        selected_columns = df.iloc[:, 2:]
        means = selected_columns.mean()
        means_list.append(means)

    df_final = pd.DataFrame({
            'Organs': ["dsc_spleen", "hd95_spleen", "dsc_right_kidney", 
                        "hd95_right_kidney", "dsc_left_kidney", "hd95_left_kidney", "dsc_stomach", 
                        "hd95_stomach", "dsc_pancreas", "hd95_pancreas", "dsc_right_adrenal_gland", 
                        "hd95_right_adrenal_gland", "dsc_left_adrenal_gland", "hd95_left_adrenal_gland", 
                        "dsc_aorta", "hd95_aorta", "dsc_inferior_vena_cava", "hd95_inferior_vena_cava", "dsc_liver", "hd95_liver"],
            'TSeg': means_list[0],
            'MRSeg': means_list[1],
            'Vibe': means_list[2]
        })
    df_final.to_csv("/data/drdcad/nicole/benchmark/organs_avg.csv", index=False)

def dice_scores_p():
    dice_list = []
    hd_list = []
    for tool in tools:
        print(tool)
        csv_files = os.listdir(f"/data/drdcad/nicole/outputs/comparison/{tool}")
        for file in csv_files:
            temp = pd.read_csv(f"/data/drdcad/nicole/outputs/comparison/{tool}/{file}")
            dice_df = temp.iloc[:, dice]
            hd_df = temp.iloc[:, hd]
            dice_series = dice_df.values.flatten()
            hd_series = hd_df.values.flatten()
            dice_list.append(dice_series)
            hd_list.append(hd_series)
    
    # Perform paired t-tests for each pair for dice
    t_stat_12, p_value_12 = ttest_rel(dice_list[0], dice_list[1])
    t_stat_13, p_value_13 = ttest_rel(dice_list[0], dice_list[2])
    t_stat_23, p_value_23 = ttest_rel(dice_list[1], dice_list[2])

    # Print results
    print("DSC")
    print(f"Pairwise Comparison 1 vs 2: T-statistic = {t_stat_12}, P-value = {p_value_12}")
    print(f"Pairwise Comparison 1 vs 3: T-statistic = {t_stat_13}, P-value = {p_value_13}")
    print(f"Pairwise Comparison 2 vs 3: T-statistic = {t_stat_23}, P-value = {p_value_23}")

    # Apply Bonferroni correction for multiple comparisons
    alpha = 0.05
    num_comparisons = 3
    p_value_corrected = min(p_value_12, p_value_13, p_value_23) * num_comparisons

    print(f"Corrected P-value (Bonferroni): {p_value_corrected}")

    # Perform paired t-tests for each pair for hd
    t_stat_12, p_value_12 = ttest_rel(hd_list[0], hd_list[1])
    t_stat_13, p_value_13 = ttest_rel(hd_list[0], hd_list[2])
    t_stat_23, p_value_23 = ttest_rel(hd_list[1], hd_list[2])

    # Print results
    print("DSC")
    print(f"Pairwise Comparison 1 vs 2: T-statistic = {t_stat_12}, P-value = {p_value_12}")
    print(f"Pairwise Comparison 1 vs 3: T-statistic = {t_stat_13}, P-value = {p_value_13}")
    print(f"Pairwise Comparison 2 vs 3: T-statistic = {t_stat_23}, P-value = {p_value_23}")

    # Apply Bonferroni correction for multiple comparisons
    alpha = 0.05
    num_comparisons = 3
    p_value_corrected = min(p_value_12, p_value_13, p_value_23) * num_comparisons

    print(f"Corrected P-value (Bonferroni): {p_value_corrected}")



#overall_boxplot()
#get_averages()
#get_organ_averages()
dice_scores_p()