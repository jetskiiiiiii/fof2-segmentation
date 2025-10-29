import os
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

table_to_stacked_bar_plot_manual()
#table_to_scatter_plot_manual()
