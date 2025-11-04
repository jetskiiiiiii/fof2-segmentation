import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
from transformation import train_transformation, eval_transformation

def table_to_scatter_plot_manual() -> None:
    """Function to convert manual foF2 identification to plots.
    Dataset can be found in dataset/data_scaling_manual
    
    Note: Specific function to be used on dataset/data_scaling_manual

    """
    table_raw_directory = "./dataset/data_scaling_manual/data_raw/"
    table_filenames = os.listdir(table_raw_directory)
    main_directory = "./dataset/data_scaling_manual"
    bar_type = "scatter_plot_with_overlay"
    split_date_by = ""

    plot_bg_color = "black"
    plot_markers_color = "white"

    for table_filename in table_filenames:
        if table_filename not in [".DS_Store", "plot_by_month", "plot_by_day"]:
            # Filenames are formatted "X Month Year-X X.csv"
            # Take out the Month and Year to get name of new file (plot we want to make)
            filename_parts = table_filename.split()
            month = filename_parts[1]
            year_with_suffix = filename_parts[2]
            year = year_with_suffix.split("-")[0]
            timeframe = f"{month}_{year}"

            filepath = os.path.join(table_raw_directory, table_filename)

            table = pd.read_csv(filepath)
            table['Day'] = table['Tgl'].ffill()
            unique_days = table['Day'].unique()
            
            for day in unique_days:
                table_day = table[table['Day'] == day].copy()
                #foF2_and_fmin_present = table_day.dropna(subset=["foF2"])
                foF2_and_fmin_present = table_day

                def time_to_decimal_hours(time_str):
                    # Handle potential NaN or missing values gracefully
                    if pd.isna(time_str):
                        return np.nan
                    H, M = map(int, time_str.split(':'))
                    return H + M / 60.0

                foF2_and_fmin_present.loc[:, 'Hour'] = foF2_and_fmin_present['UT+7'].apply(time_to_decimal_hours)

                scatter_data = {
                    "x": foF2_and_fmin_present["Hour"],
                    "ymin": foF2_and_fmin_present["fmin"],
                    "ymax": foF2_and_fmin_present["foF2"]
                }

                savepath = f"{main_directory}/{bar_type}/{split_date_by}/{timeframe}_{day}.png"
                date = {"day": int(day), "month": month, "year": int(year)}

                get_fti_and_overlay_mask(
                    savepath=savepath,
                    date=date,
                    scatter_data = scatter_data,
                    plot_markers_color=plot_markers_color,
                    plot_bg_color=plot_bg_color,
                    plot_type="scatter"
                )

def table_to_stacked_bar_plot_manual() -> None:
    """Function to convert manual foF2 identification to plots.
    Dataset can be found in dataset/data_scaling_manual
    
    Note: Specific function to be used on dataset/data_scaling_manual

    """
    table_raw_directory = "./dataset/data_scaling_manual/data_raw/"
    table_filenames = os.listdir(table_raw_directory)
    main_directory = "./dataset/data_scaling_manual"
    bar_type = "stacked_bar_plot_with_overlay"
    split_date_by = ""

    plot_bg_color = "black"
    plot_markers_color = "white"

    for table_filename in table_filenames:
        if table_filename not in [".DS_Store", "plot_by_month", "plot_by_day"]:
            # Filenames are formatted "X Month Year-X X.csv"
            # Take out the Month and Year to get name of new file (plot we want to make)
            filename_parts = table_filename.split()
            month = filename_parts[1]
            year_with_suffix = filename_parts[2]
            year = year_with_suffix.split("-")[0]
            timeframe = f"{month}_{year}"

            filepath = os.path.join(table_raw_directory, table_filename)

            table = pd.read_csv(filepath)
            table['Day'] = table['Tgl'].ffill()
            unique_days = table['Day'].unique()
            
            for day in unique_days:
                table_day = table[table['Day'] == day].copy()
                #foF2_and_fmin_present = table_day.dropna(subset=["foF2"])
                foF2_and_fmin_present = table_day

                def time_to_decimal_hours(time_str):
                    # Handle potential NaN or missing values gracefully
                    if pd.isna(time_str):
                        return np.nan
                    H, M = map(int, time_str.split(':'))
                    return H + M / 60.0

                foF2_and_fmin_present.loc[:, 'Hour'] = foF2_and_fmin_present['UT+7'].apply(time_to_decimal_hours)

                scatter_data = {
                    "x": foF2_and_fmin_present["Hour"],
                    "ymin": foF2_and_fmin_present["fmin"],
                    "ymax": foF2_and_fmin_present["foF2"]
                }

                savepath = f"{main_directory}/{bar_type}/{split_date_by}/{timeframe}_{day}.png"
                date = {"day": int(day), "month": month, "year": int(year)}

                get_fti_and_overlay_mask(
                    savepath=savepath,
                    date=date,
                    scatter_data=scatter_data,
                    plot_markers_color=plot_markers_color,
                    plot_bg_color=plot_bg_color,
                    plot_type="stacked_bar"
                )


def get_fti_and_overlay_mask(savepath, date, scatter_data, plot_markers_color, plot_bg_color, plot_type):
    """ Date: day (int), month (str, 1st 3 letters of month), year (int)

    """
    day = date["day"]
    month = date["month"]
    year = date["year"]

    main_directory = "./dataset/data_scaling_manual/data_fti/"
    main_path = Path(main_directory)

    month_path = ""
    for folder in main_path.glob(f"*LTPMP-{month}-{str(year)}"):
        if folder.is_dir():
            month_path = str(folder.resolve())

    month_path = Path(month_path)
    date_path = ""
    for file in month_path.glob(f"*{str(day)}-{month}-{str(year)}.png"):
        if file.is_file():
            date_path = Path(str(file.resolve()))

    # Sometimes data in csv are for days that don't exist in that year
    if date_path == "":
        return None

    image = cv.imread(date_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    transformed = eval_transformation(image=image_rgb)
    transformed_image = transformed["image"]
    img_height, img_width = transformed_image.shape[:2]

    dpi = 100
    alpha = 0.4
    fig_dim = 640 / dpi 
    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)

    ax.set_xlim(0, 24)  
    ax.set_ylim(0, 20)
    
    # Plot image
    ax.imshow(
        transformed_image,
        extent=[0, 24, 0, 20],
        aspect="auto",
        zorder=0
    )

    if plot_type == "scatter":

        ax.vlines(
            x=scatter_data["x"],
            ymin=scatter_data["ymin"],
            ymax=scatter_data["ymax"],
            colors=plot_markers_color,
            linestyle='-',
            linewidth=8,
            alpha=alpha,
            zorder=2
        )
        
        # Plot 1: fmin markers
        ax.scatter(
            scatter_data["x"],
            scatter_data["ymin"],
            color=plot_markers_color,
            marker='o',
            alpha=alpha,
            zorder=2
        )

        # Plot 2: foF2 markers
        ax.scatter(
            scatter_data["x"],
            scatter_data["ymax"],
            color=plot_markers_color,
            marker='o',
            alpha=alpha,
            zorder=2
        )

    elif plot_type == "stacked_bar":
        length = len(scatter_data["x"])
        bottom = np.full(length, 2)
        width = 0.25
        for row_idx in range(length):
            ax.bar(
                scatter_data["x"].iloc[row_idx],
                scatter_data["ymin"].iloc[row_idx],
                width,
                bottom=bottom,
                color=plot_bg_color,
                alpha=0,
            )

            ax.bar(
                scatter_data["x"].iloc[row_idx],
                scatter_data["ymax"].iloc[row_idx],
                width,
                bottom=scatter_data["ymin"].iloc[row_idx],
                color=plot_markers_color,
                alpha=0.4,
            )


    # Get rid of borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    #fig.patch.set_alpha(0.0) # Makes the figure background transparent
    #ax.patch.set_alpha(0.0)

    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_ylim(bottom=0)
    plt.tight_layout()
    #plt.show()

    fig.savefig(savepath)
    plt.close()

def separate_month_into_days():
    """
    Original manual data is compiled of all days in month.
    This function separates those days into their own CSVs.
    """
    path_to_save_days_df = "./dataset/data_scaling_manual/data_raw/days"
    manual_data_path = "./dataset/data_scaling_manual/data_raw/"
    manual_data_filenames = os.listdir(manual_data_path)
    for filename in manual_data_filenames:
        split_filename = re.split(r"[ -]", filename)
        if len(split_filename) != 5:
            continue
        year, month = int(split_filename[2]), split_filename[1]

        path_to_monthly_data = os.path.join(manual_data_path, filename)
        month_df = pd.read_csv(path_to_monthly_data)

        #print(True if len(month_df) % 96 == 0 else False)
        for i in range(0, len(month_df), 96):
            day_df = month_df.loc[i:i+95, ["UT+7", "fmin", "foES", "foF2", "Spread-F", "h'F"]]
            if len(day_df) % 96 == 0:
                # Calling reset_index twice to first discard old indexing from original and then to convert new created index into a data column
                day_df = day_df.reset_index(drop=True)
                day_df = day_df.reset_index(drop=False)
                day_df.to_csv(f"{path_to_save_days_df}/{i//96+1}-{month}-{year}.csv", index=False)

def prepare_manual_and_numeric_for_evaluation(path_to_numeric_csv_dir: str, path_to_manual_dir: str, path_to_save_numeric_csv_dir: str, path_to_save_manual_dir: str):
    """
    Takes the original manual and numeric files and deals with missing foES, foF2 and fmin.
    Also adds combined foES_foF2 in manual.
    """
    numeric_filenames = os.listdir(path_to_numeric_csv_dir)

    # Need to loop through entire dataset twice, first to calculate global_mean_manual then to calculate final RSE
    for numeric_filename in numeric_filenames:
        split = re.split(r"[.-]", numeric_filename)
        if split[3] != "2020":
            continue
        manual_filename = f"{path_to_manual_dir}/days/{split[1]}-{split[2]}-{split[3]}.csv"

        numeric_df = pd.read_csv(os.path.join(path_to_numeric_csv_dir, numeric_filename))
        manual_df = pd.read_csv(manual_filename)

        # Deal with missing foES, foF2
        missing_foes_and_fof2_manual = manual_df[["foES", "foF2"]].isna().all(axis=1)
        missing_foes_and_fof2_numeric = numeric_df["foF2"].isna()
        manual_df = manual_df[~missing_foes_and_fof2_manual & ~missing_foes_and_fof2_numeric]
        numeric_df = numeric_df[~missing_foes_and_fof2_manual & ~missing_foes_and_fof2_numeric]

        # Deal with missing fmin
        missing_fmin_manual = manual_df["fmin"].isna()
        missing_fmin_numeric = numeric_df["fmin"].isna()
        manual_df = manual_df[~missing_fmin_manual & ~missing_fmin_numeric]
        numeric_df = numeric_df[~missing_fmin_manual & ~missing_fmin_numeric]

        assert len(manual_df) == len(numeric_df), "Error in filtering either manual or numeric."

        # Get either foF2 or foES from manual, preferring foF2 if exists
        manual_df["foES_foF2"] = np.where(manual_df["foF2"].notna(), manual_df["foF2"], manual_df["foES"])

        manual_df.columns.values[0] = "index"
        numeric_df.columns.values[0] = "index"
        manual_df = manual_df.reset_index(drop=True)
        numeric_df = numeric_df.reset_index(drop=True)

        manual_df.to_csv(f"{path_to_save_manual_dir}/{numeric_filename}", index=False)
        numeric_df.to_csv(f"{path_to_save_numeric_csv_dir}/{numeric_filename}", index=False)


def get_metrics_all_numeric_with_manual(path_to_numeric_csv_dir: str, path_to_manual_dir: str):
    """
    """
    numeric_filenames = os.listdir(path_to_numeric_csv_dir)

    global_top_rse, global_bot_rse, global_total_values, global_sum_manual, global_mean_manual = 0, 0, 0, 0, 0
    global_mse = 0
    final_rse, final_rmse = 0, 0
    target_range = {"min": 100000, "max": 0}    # Setting min to arbitrarily high value to compare with real mins
    final_range, global_min, global_max = 0, 0, 0
    # Need to loop through entire dataset twice, first to calculate global_mean_manual then to calculate final RSE
    for i in range(2):
        for numeric_filename in numeric_filenames:
            # Ignore .DS_Store
            if numeric_filename.startswith("."):
                continue
            numeric_path = os.path.join(path_to_numeric_csv_dir, numeric_filename)
            numeric_df = pd.read_csv(numeric_path)
            manual_df = pd.read_csv(os.path.join(path_to_manual_dir, numeric_filename))

            # For first iteration, we can only update total values, sum of values, and top RSE
            if i == 0:
                # Get total and sum of values in ground truth (manual) of current test item
                current_total_values = 2*len(manual_df)
                current_sum_manual = manual_df["fmin"].sum() + manual_df["foES_foF2"].sum()

                # Update global values
                global_total_values += current_total_values
                global_sum_manual += current_sum_manual

                # Calculate numerator of RSE and update global value
                current_top_rse = (
                    (manual_df["fmin"] - numeric_df["fmin"])**2 +
                    (manual_df["foES_foF2"] - numeric_df["foF2"])**2
                ).sum()
                global_top_rse += current_top_rse

                # Calculating range to get context of RMSE
                # We can assume fmin will always contain the absolute min and foES/foF2 will contain absolute max values
                current_min = numeric_df["fmin"].min()
                current_max = numeric_df["foF2"].max()
                target_range["min"] = current_min if current_min < target_range["min"] else target_range["min"]
                target_range["max"] = current_max if current_max > target_range["max"] else target_range["max"]

            # In the second iteration, we have global mean, so we can calculate bot RSE
            else:
                current_bot_rse = (
                    (manual_df["fmin"] - global_mean_manual)**2 +
                    (manual_df["foES_foF2"] - global_mean_manual)**2
                ).sum()
                global_bot_rse += current_bot_rse

                # Also calculate MSE in 2nd iteration
                current_mse = (
                    (manual_df["fmin"] - numeric_df["fmin"])**2 +
                    (manual_df["foES_foF2"] - numeric_df["foF2"])**2
                ).sum()
                global_mse += current_mse

        # After first iteration, sum and total will be calculated for all test items
        if i == 0:
            # We can now update global mean
            global_mean_manual = global_sum_manual / global_total_values
        # In the second iteration, we can calculate final RSE
        else:
            # Calculating RMSE
            final_rse = global_top_rse / global_bot_rse
            final_rmse = np.sqrt(global_mse / global_total_values)
            global_min = target_range["min"]
            global_max = target_range["max"]
            final_range = global_max - global_min

    return round(final_rse, 3), round(final_rmse, 3), round(global_min, 3), round(global_max, 3), round(final_range, 3)

def get_metrics_single_numeric_with_manual(path_to_numeric_csv: str, path_to_manual_csv: str):
    numeric_df = pd.read_csv(path_to_numeric_csv)
    manual_df = pd.read_csv(path_to_manual_csv)

    total_values = 2*len(manual_df)

    # Calculating RSE
    mean_manual = (manual_df["fmin"].sum() + manual_df["foES_foF2"].sum()) / total_values
    top_rse, bot_rse = 0, 0
    top_rse = (
        (manual_df["fmin"] - numeric_df["fmin"])**2 +
        (manual_df["foES_foF2"] - numeric_df["foF2"])**2
    ).sum()
    bot_rse = (
        (manual_df["fmin"] - mean_manual)**2 +
        (manual_df["foES_foF2"] - mean_manual)**2
    ).sum()
    rse = top_rse / bot_rse

    # Calculating RMSE
    mse = (
        (manual_df["fmin"] - numeric_df["fmin"])**2 +
        (manual_df["foES_foF2"] - numeric_df["foF2"])**2
    ).sum() / total_values
    rmse = np.sqrt(mse)

    gmin = numeric_df["fmin"].min()
    gmax = numeric_df["foF2"].max()
    grange = gmax-gmin

    return round(rse, 3), round(rmse, 3), round(gmin, 3), round(gmax, 3), round(grange, 3)

def get_metrics_all_quickscale_with_manual(path_to_manual_days_dir: str, path_to_quickscale_dir: str):
    """
    """
    quickscale_filenames = os.listdir(path_to_quickscale_dir)

    global_top_rse, global_bot_rse, global_total_values, global_sum_manual, global_mean_manual = 0, 0, 0, 0, 0
    global_mse = 0
    final_rse, final_rmse = 0, 0
    target_range = {"min": 100000, "max": 0}    # Setting min to arbitrarily high value to compare with real mins
    final_range, global_min, global_max = 0, 0, 0

    # Need to loop through entire dataset twice, first to calculate global_mean_manual then to calculate final RSE
    for i in range(2):
        for quickscale_filename in quickscale_filenames:
            # Ignore .DS_Store
            if quickscale_filename.startswith("."):
                continue
            qs_df = pd.read_csv(os.path.join(path_to_quickscale_dir, quickscale_filename))
            manual_df = pd.read_csv(os.path.join(path_to_manual_days_dir, quickscale_filename)) # Using quickscale_filename because filenames are same

            # Add time (float) column to manual
            manual_df["time_as_float"] = np.arange(0.0, 24, 0.25) 

            # Split qs into separate df for fmin/foF2 (because times don't match up)
            qs_fmin = qs_df[qs_df["Parameter"] == "fmin"]
            qs_fof2 = qs_df[qs_df["Parameter"] == "foF2"]
            
            # Ensure values are sorted before merging
            qs_fmin = qs_fmin.sort_values("JamDec").reset_index()
            qs_fof2 = qs_fmin.sort_values("JamDec").reset_index()
            
            # Possible that merged JamDec of fmin and fof2 are same
            merged_fmin_df = pd.merge_asof(
                manual_df,
                qs_fmin,
                left_on="time_as_float",
                right_on="JamDec",
                direction="nearest",
                suffixes=('_A', '_B')
            )
            merged_fof2_df = pd.merge_asof(
                manual_df,
               qs_fof2,
                left_on="time_as_float",
                right_on="JamDec",
                direction="nearest",
                suffixes=('_A', '_B')
            )

            # Deal with missing foES, foF2
            missing_foes_and_fof2_manual = merged_fof2_df[["foES", "foF2"]].isna().all(axis=1)
            merged_fof2_df = merged_fof2_df[~missing_foes_and_fof2_manual]
            merged_fof2_df["foES_foF2"] = np.where(merged_fof2_df["foF2"].notna(), merged_fof2_df["foF2"], merged_fof2_df["foES"]) # Create a column pulling from foF2 or fmin

            # Deal with missing fmin
            missing_fmin_manual = merged_fmin_df["fmin"].isna()
            merged_fmin_df = merged_fmin_df[~missing_fmin_manual]

            # For first iteration, we can only update total values, sum of values, and top RSE
            if i == 0:
                # Get total and sum of values in ground truth (manual) of current test item
                current_total_values = len(merged_fmin_df)+len(merged_fof2_df)
                current_sum_manual = merged_fmin_df["fmin"].sum() + merged_fof2_df["foES_foF2"].sum()

                # Update global values
                global_total_values += current_total_values
                global_sum_manual += current_sum_manual

                # Calculate numerator of RSE and update global value
                current_top_rse = (
                    (merged_fmin_df["fmin"] - merged_fmin_df["Nilai"])**2 +
                    (merged_fof2_df["foES_foF2"] - merged_fof2_df["Nilai"])**2
                ).sum()
                global_top_rse += current_top_rse

                # Calculating range to get context of RMSE
                # We can assume fmin will always contain the absolute min and foES/foF2 will contain absolute max values
                current_min = merged_fmin_df["Nilai"].min()
                current_max = merged_fof2_df["Nilai"].max()
                target_range["min"] = current_min if current_min < target_range["min"] else target_range["min"]
                target_range["max"] = current_max if current_max > target_range["max"] else target_range["max"]

            # In the second iteration, we have global mean, so we can calculate bot RSE
            else:
                current_bot_rse = (
                    (merged_fmin_df["fmin"] - global_mean_manual)**2 +
                    (merged_fof2_df["foES_foF2"] - global_mean_manual)**2
                ).sum()
                global_bot_rse += current_bot_rse

                # Also calculate MSE in 2nd iteration
                current_mse = (
                    (merged_fmin_df["fmin"] - merged_fmin_df["Nilai"])**2 +
                    (merged_fof2_df["foES_foF2"] - merged_fof2_df["Nilai"])**2
                ).sum()
                global_mse += current_mse

        # After first iteration, sum and total will be calculated for all test items
        if i == 0:
            # We can now update global mean
            global_mean_manual = global_sum_manual / global_total_values
        # In the second iteration, we can calculate final RSE
        else:
            # Calculating RMSE
            final_rse = global_top_rse / global_bot_rse
            final_rmse = np.sqrt(global_mse / global_total_values)
            global_min = target_range["min"]
            global_max = target_range["max"]
            final_range = global_max - global_min

    return round(final_rse, 3), round(final_rmse, 3), round(global_min, 3), round(global_max, 3), round(final_range, 3)


def get_metrics_single_quickscale_with_manual(path_to_manual_csv: str, path_to_quickscale_csv: str):
    """
    1. split qs into fmin and fof2
    2. merge manual with fmin and fof2 (separately but possible that JamDec values are same)
    """
    manual_df = pd.read_csv(path_to_manual_csv)
    qs_df = pd.read_csv(path_to_quickscale_csv)

    # Add time (float) column to manual
    manual_df["time_as_float"] = np.arange(0.0, 24, 0.25) 

    # Split qs into separate df for fmin/foF2 (because times don't match up)
    qs_fmin = qs_df[qs_df["Parameter"] == "fmin"]
    qs_fof2 = qs_df[qs_df["Parameter"] == "foF2"]
    
    # Ensure values are sorted before merging
    qs_fmin = qs_fmin.sort_values("JamDec").reset_index()
    qs_fof2 = qs_fmin.sort_values("JamDec").reset_index()
    
    # Possible that merged JamDec of fmin and fof2 are same
    merged_fmin_df = pd.merge_asof(
        manual_df,
        qs_fmin,
        left_on="time_as_float",
        right_on="JamDec",
        direction="nearest",
        suffixes=('_A', '_B')
    )
    merged_fof2_df = pd.merge_asof(
        manual_df,
        qs_fof2,
        left_on="time_as_float",
        right_on="JamDec",
        direction="nearest",
        suffixes=('_A', '_B')
    )

    # Deal with missing foES, foF2
    missing_foes_and_fof2_manual = merged_fof2_df[["foES", "foF2"]].isna().all(axis=1)
    merged_fof2_df = merged_fof2_df[~missing_foes_and_fof2_manual]
    merged_fof2_df["foES_foF2"] = np.where(merged_fof2_df["foF2"].notna(), merged_fof2_df["foF2"], merged_fof2_df["foES"]) # Create a column pulling from foF2 or fmin

    # Deal with missing fmin
    missing_fmin_manual = merged_fmin_df["fmin"].isna()
    merged_fmin_df = merged_fmin_df[~missing_fmin_manual]

    total_values = len(merged_fmin_df)+len(merged_fof2_df)

    # Calculating RSE
    # We pull from the same df, with ground truth column being "fmin/foF2/foES" and predicted being "Nilai"
    mean_manual = (merged_fmin_df["fmin"].sum() + merged_fof2_df["foF2"].sum()) / total_values
    top_rse, bot_rse = 0, 0
    top_rse = (
        (merged_fmin_df["fmin"] - merged_fmin_df["Nilai"])**2 +
        (merged_fof2_df["foES_foF2"] - merged_fof2_df["Nilai"])**2
    ).sum()
    bot_rse = (
        (merged_fmin_df["fmin"] - mean_manual)**2 +
        (merged_fof2_df["foES_foF2"] - mean_manual)**2
    ).sum()
    rse = top_rse / bot_rse

    # Calculating RMSE
    mse = (
        (merged_fmin_df["fmin"] - merged_fmin_df["Nilai"])**2 +
        (merged_fof2_df["foES_foF2"] - merged_fof2_df["Nilai"])**2
    ).sum() / total_values
    rmse = np.sqrt(mse)

    gmin = merged_fmin_df["Nilai"].min()
    gmax = merged_fof2_df["Nilai"].max()
    grange = gmax-gmin

    return round(rse, 3), round(rmse, 3), round(gmin, 3), round(gmax, 3), round(grange, 3)

if __name__ == "__main__":
    ## Turn manual into plots/masks
    #table_to_stacked_bar_plot_manual()
    #table_to_scatter_plot_manual()

    ## Original manual data was grouped into months
    #separate_month_into_days()

    ## Get eval metrics between numeric and manual
    #version = "v22"
    #path_to_numeric_csv_dir = f"./predictions/numeric_csv/original/{version}"
    path_to_manual_dir = "./dataset/data_scaling_manual/data_raw"
    #path_to_save_numeric_csv_dir = f"./predictions/numeric_csv/prepared_for_numeric_eval/{version}"
    #path_to_save_manual_dir = f"./dataset/data_scaling_manual/data_raw/prepared_for_numeric_eval/{version}"
    ## Preparing numeric and manual
    #prepare_manual_and_numeric_for_evaluation(path_to_numeric_csv_dir, path_to_manual_dir, path_to_save_numeric_csv_dir, path_to_save_manual_dir)
    ## Getting metrics
    #RSE, RMSE, gmin, gmax, grange = get_metrics_all_numeric_with_manual(path_to_save_numeric_csv_dir, path_to_save_manual_dir)
    #print(RSE, RMSE, gmin, gmax, grange)

    # Get eval metrics between quickscale and manual
    path_to_quickscale_dir = "./dataset/data_pak_jiyo"
    path_to_manual_days_dir = "./dataset/data_scaling_manual/data_raw/days"
    RSE, RMSE, gmin, gmax, grange = get_metrics_all_quickscale_with_manual(path_to_manual_days_dir, path_to_quickscale_dir)
    print(RSE, RMSE, gmin, gmax, grange)
