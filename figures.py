import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.stats import probplot
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
# import statsmodels sm,ols and pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


tools = ["ts", "mr", "vibe"]
dice = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
hd = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

plt.rcParams.update({
    'font.size': 20,          # Font size for all text elements
    'axes.titlesize': 30,     # Font size for axis titles
    'axes.labelsize': 25,     # Font size for axis labels
    'xtick.labelsize': 25,    # Font size for x-axis tick labels
    'ytick.labelsize': 25,    # Font size for y-axis tick labels
    'legend.fontsize': 20,    # Font size for legend
    'figure.titlesize': 35,   # Font size for figure title
})


def overall_boxplot():

    dice_series_list = []
    hd_series_list = []
    tool_labels = []


    for tool in tools:

        csv_files = os.listdir(f"/Users/nicol/Documents/nih/outputs/{tool}")

        df_list = [pd.read_csv(f"/Users/nicol/Documents/nih/outputs/{tool}/{file}") for file in csv_files if file.endswith(".csv")]
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
    plt.title('Dice Score Distribution')
    plt.xticks(ticks=range(len(tools)), labels=["TS", "MRSeg", "Vibe"])
    plt.xlabel('')
    plt.ylabel('DSC')
    
    # Boxplot for hd series
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Tool', y='Value', data=hd_df_final, palette=palette)
    plt.title('HD Error Distribution')
    plt.xticks(ticks=range(len(tools)), labels=["TS", "MRSeg", "Vibe"])
    plt.xlabel('')
    plt.ylabel('HD (mm)')


    # Adjust layout
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('comparison_boxplots.pdf')  # You can specify the file name and format (e.g., .png, .pdf)

    # Close the plot to free up memory
    plt.close()

def get_averages():
    
    for tool in tools:
        print(tool)
        csv_files = [file for file in os.listdir(f"/Users/nicol/Documents/nih/outputs/{tool}") if file.endswith(".csv")]
        all_dsc = []
        all_hd = []
        for file in csv_files:
            temp = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/{tool}/{file}")
            dice_df = temp.iloc[:, dice]
            hd_df = temp.iloc[:, hd]
            dice_series = dice_df.values.flatten()
            hd_series = hd_df.values.flatten()
            
            # Identify indices where hd_series is not infinite
            valid_indices = ~np.isinf(hd_series)

            # Filter out the rows where hd_series is infinite
            filtered_dice_series = dice_series[valid_indices]
            filtered_hd_series = hd_series[valid_indices]
            
            all_dsc.append(filtered_dice_series)
            all_hd.append(filtered_hd_series)
            
            # Calculate the means
            dice_mean = filtered_dice_series.mean()
            hd_mean = filtered_hd_series.mean()
            
            # Calculate the standard deviations
            dice_std = filtered_dice_series.std()
            hd_std = filtered_hd_series.std()

            # Print the results
            print(f"Dice score for {file}: {dice_mean} (Std: {dice_std})")
            print(f"HD for {file}: {hd_mean} (Std: {hd_std})")
        combined_dsc = np.concatenate(all_dsc)
        combined_hd = np.concatenate(all_hd)
        print(f"Dice score for {tool}: {combined_dsc.mean()} (Std: {combined_dsc.std()})")
        print(f"HD for {tool}: {combined_hd.mean()} (Std: {combined_hd.std()})")

def get_organ_averages():
    means_list = []
    for tool in tools:
        print(tool)
        csv_files = [file for file in os.listdir(f"/Users/nicol/Documents/nih/outputs/{tool}") if file.endswith(".csv")]
        df_list = [pd.read_csv(f"/Users/nicol/Documents/nih/outputs/{tool}/{file}") for file in csv_files]
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
    df_final.to_csv("/Users/nicol/Documents/nih/outputs/organs_avg.csv", index=False)

def dice_scores_p():
    dice_list = []
    hd_list = []
    for tool in tools:
        print(tool)
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

    
    # Clean combined_dice: remove 0 values
    dice_list = [series[series != 0] for series in combined_dice]

    print(dice_list[0].mean())
    print(dice_list[1].mean())
    print(dice_list[2].mean())
    probplot(dice_list[0], dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()
    probplot(dice_list[2], dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()
    probplot(dice_list[1], dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    plt.show()

    # Clean combined_hd: remove inf values
    hd_list = [series[np.isfinite(series)] for series in combined_hd]
    print(hd_list)

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
    _, p_value_corrected, _, _ = multipletests([p_value_12, p_value_13, p_value_23], alpha=0.05, method='bonferroni')

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
    #p_value_corrected = min(p_value_12, p_value_13, p_value_23) * num_comparisons
    _, p_value_corrected, _, _ = multipletests([p_value_12, p_value_13, p_value_23], alpha=0.05, method='bonferroni')


    print(f"Corrected P-value (Bonferroni): {p_value_corrected}")

def dice_scores_sequence(tool):
    #g1 delayed, g2 arterial, g3 precontrast, g4 venous
    dsc_groups = []
    hd_groups = []
    
    csv_files = [file for file in os.listdir(f"/Users/nicol/Documents/nih/outputs/{tool}") if file.endswith(".csv")]
    
    for file in csv_files:
        temp = pd.read_csv(f"/Users/nicol/Documents/nih/outputs/{tool}/{file}")
        dice_df = temp.iloc[:, dice]
        hd_df = temp.iloc[:, hd]
        dice_series = dice_df.values.flatten()
        hd_series = hd_df.values.flatten()
        
        # Identify indices where hd_series is not infinite
        valid_indices = ~np.isinf(hd_series)

        # Filter out the rows where hd_series is infinite
        filtered_dice_series = dice_series[valid_indices]
        filtered_hd_series = hd_series[valid_indices]
        probplot(filtered_dice_series, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        plt.show()
        probplot(filtered_hd_series, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        plt.show()
        
        dsc_groups.append(filtered_dice_series)
        hd_groups.append(filtered_hd_series)
    
    for groups in [dsc_groups, hd_groups]:
        p_values = []
        pairs = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                stat, p_value = ttest_ind(groups[i], groups[j])
                p_values.append(p_value)
                pairs.append((i+1, j+1))

        # Display the results
        pairwise_results = pd.DataFrame({
            'Comparison': [f'Group {pair[0]} vs Group {pair[1]}' for pair in pairs],
            'p-value': p_values
        })

        # Apply Bonferroni correction
        _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

        # Add the corrected p-values to the DataFrame
        pairwise_results['Corrected p-value'] = corrected_p_values

        print(pairwise_results)

def all_boxplot():
    # Load data from the three sheets
    df_ts = pd.read_csv("/Users/nicol/Documents/nih/outputs/ts/venous.csv")
    for i in ("arterial", "precontrast", "delayed"):
        df_ts = pd.concat([df_ts, pd.read_csv(f"/Users/nicol/Documents/nih/outputs/ts/{i}.csv")])
    df_mr = pd.read_csv("/Users/nicol/Documents/nih/outputs/mr/venous.csv")
    for i in ("arterial", "precontrast", "delayed"):
        df_mr = pd.concat([df_mr, pd.read_csv(f"/Users/nicol/Documents/nih/outputs/mr/{i}.csv")])
    df_vibe = pd.read_csv("/Users/nicol/Documents/nih/outputs/vibe/venous.csv")
    for i in ("arterial", "precontrast", "delayed"):
        df_vibe = pd.concat([df_vibe, pd.read_csv(f"/Users/nicol/Documents/nih/outputs/vibe/{i}.csv")])
    

    # Add a column to identify the tool
    df_ts['Tool'] = 'TS'
    df_mr['Tool'] = 'MRSeg'
    df_vibe['Tool'] = 'VIBE'

    # Combine the data from all tools
    combined_df = pd.concat([df_ts, df_mr, df_vibe])

    # Prepare the data for plotting
    # Melt the DataFrame to long format
    dsc_columns = [col for col in combined_df.columns if col.startswith('dsc')]
    hd_columns = [col for col in combined_df.columns if col.startswith('hd')]

    # Melt Dice Score data
    dsc_long_df = combined_df.melt(id_vars=['Tool'], value_vars=dsc_columns, var_name='Organ', value_name='Dice Score')
    dsc_long_df['Organ'] = dsc_long_df['Organ'].str.split('_', n=1, expand=True)[1]  # Extract organ name

    # Melt HD data
    hd_long_df = combined_df.melt(id_vars=['Tool'], value_vars=hd_columns, var_name='Organ', value_name='HD')
    hd_long_df['Organ'] = hd_long_df['Organ'].str.split('_', n=1, expand=True)[1]  # Extract organ name

    # Define organ categories
    large_organs = ['liver', 'spleen', 'stomach']
    medium_organs = ['right_kidney', 'left_kidney', 'pancreas']
    small_organs = ['right_adrenal_gland', 'left_adrenal_gland', 'inferior_vena_cava', 'aorta']

    # Filter data for each organ category
    dsc_large_df = dsc_long_df[dsc_long_df['Organ'].isin(large_organs)]
    dsc_medium_df = dsc_long_df[dsc_long_df['Organ'].isin(medium_organs)]
    dsc_small_df = dsc_long_df[dsc_long_df['Organ'].isin(small_organs)]

    hd_large_df = hd_long_df[hd_long_df['Organ'].isin(large_organs)]
    hd_medium_df = hd_long_df[hd_long_df['Organ'].isin(medium_organs)]
    hd_small_df = hd_long_df[hd_long_df['Organ'].isin(small_organs)]

    # Create boxplots for each category
    def create_boxplot(data, value_col, title, filename, small=False):
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=data, x='Organ', y=value_col, hue='Tool')
        plt.title(title)
        plt.tight_layout()
        if small:
            plt.xticks(fontsize = 15)
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(filename)
        plt.close()
        

    # Boxplots for large organs
    create_boxplot(dsc_large_df, 'Dice Score', 'Dice Scores for Large Organs', 'dsc_large_organs.pdf')
    create_boxplot(hd_large_df, 'HD', 'HD Values for Large Organs', 'hd_large_organs.pdf')

    # Boxplots for medium organs
    create_boxplot(dsc_medium_df, 'Dice Score', 'Dice Scores for Medium Organs', 'dsc_medium_organs.pdf')
    create_boxplot(hd_medium_df, 'HD', 'HD Values for Medium Organs', 'hd_medium_organs.pdf')

    # Boxplots for small organs
    create_boxplot(dsc_small_df, 'Dice Score', 'Dice Scores for Small Organs', 'dsc_small_organs.pdf', small=True)
    create_boxplot(hd_small_df, 'HD', 'HD Values for Small Organs', 'hd_small_organs.pdf', small=True)

def wilxcom():
    dice_list = []
    hd_list = []
    for tool in tools:
        print(tool)
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

    
    # Clean combined_dice: remove 0 values
    dice_list = [series[series != 0] for series in combined_dice]


    print(dice_list[0].mean())
    print(dice_list[1].mean())
    print(dice_list[2].mean())
    
    group1 = dice_list[0]
    group2 = dice_list[1]
    group3 = dice_list[2]

    # Perform Wilcoxon signed-rank tests for each pair
    p_values = []
    p_values.append(wilcoxon(group1, group2)[1])
    p_values.append(wilcoxon(group1, group3)[1])
    p_values.append(wilcoxon(group2, group3)[1])

    # Apply Bonferroni correction
    corrected_p_values = multipletests(p_values, alpha=0.05, method='bonferroni')[1]

    print(f"Corrected p-values: {corrected_p_values}")

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

def turkey(sequence):
    print(sequence)
    # save the state of randomness
    np.random.seed(123)
    
   
    
    # dsc
    tseg = get_data("ts", sequence)[0]

    mrseg = get_data("mr", sequence)[0]

    vibe = get_data("vibe", sequence)[0]

    
    # print out the individual cookie weights for each bakery
    #print(f'tseg: {tseg}\nmrseg: {mrseg}\nvibe: {vibe}')

    print("dsc")
    # incorporate bakery_id and cookie weights into pandas DataFrame
    # where cookies arrays are sequentially concatenated into a single array
    dsc_df = pd.DataFrame({'tool' : ['tseg']*len(tseg) + ['mrseg']*len(tseg) + ['vibe']*len(tseg),
                                'dsc' : np.concatenate((tseg,
                                                            mrseg,
                                                            vibe),axis=0)})

    # create a model with statsmodels ols and fit data into the model
    model = ols(formula='dsc ~ tool', data=dsc_df).fit()

    # call statsmodels stats.anova_lm and pass model as an argument to perform ANOVA
    anova_result = sm.stats.anova_lm(model, type=2)

    print(anova_result)
    multiple_comp_result = pairwise_tukeyhsd(endog=dsc_df['dsc'],
                                         groups=dsc_df['tool'],
                                        alpha=0.05)
    print(multiple_comp_result.summary())
    
    ##################
    
    # hd
    tseg = get_data("ts", sequence)[1]
    mrseg = get_data("mr", sequence)[1]
    vibe = get_data("vibe", sequence)[1]
    

    print("hd")
    # incorporate bakery_id and cookie weights into pandas DataFrame
    # where cookies arrays are sequentially concatenated into a single array
    hd_df = pd.DataFrame({'tool' : ['tseg']*len(tseg) + ['mrseg']*len(tseg) + ['vibe']*len(tseg),
                                'hd' : np.concatenate((tseg,
                                                            mrseg,
                                                            vibe),axis=0)})

    # create a model with statsmodels ols and fit data into the model
    model = ols(formula='hd ~ tool', data=hd_df).fit()

    # call statsmodels stats.anova_lm and pass model as an argument to perform ANOVA
    anova_result = sm.stats.anova_lm(model, type=2)

    print(anova_result)
    multiple_comp_result = pairwise_tukeyhsd(endog=hd_df['hd'],
                                         groups=hd_df['tool'],
                                        alpha=0.05)
    print(multiple_comp_result.summary())

#overall_boxplot()
get_averages()
#get_organ_averages()
#dice_scores_p()
#all_boxplot()
#dice_scores_sequence("vibe")
#wilxcom()
#for sequence in ["precontrast","arterial","delayed", "venous"]:
#    turkey(sequence)