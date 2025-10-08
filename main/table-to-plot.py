import os
import pandas as pd
import matplotlib.pyplot as plt

def table_to_plot_manual() -> None:
    """Function to convert manual foF2 identification to plots.
    Dataset can be found in dataset/data_scaling_manual
    
    Note: Specific function to be used on dataset/data_scaling_manual

    """
    table_raw_directory = "./dataset/data_scaling_manual/data_raw/"
    table_filenames = os.listdir(table_raw_directory)
    main_directory = "./dataset/data_scaling_manual/"
    split_date_by = "plot_by_day"

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
                foF2_and_fmin_present = table_day.dropna(subset=["foF2"])

                dpi = 100
                fig_dim = 640 / dpi 

                plt.figure(figsize=(fig_dim, fig_dim))
                plt.ylim(0, 20)

                # Plot 0: Vertical connecting lines
                plt.vlines(
                    x=foF2_and_fmin_present["UT+7"],
                    ymin=foF2_and_fmin_present["fmin"],
                    ymax=foF2_and_fmin_present["foF2"],
                    colors=plot_markers_color,
                    linestyle='-',
                    linewidth=8,
                    alpha=1,
                    #label='A to B Connection'
                )

                # Plot 1: Column B vs C
                # Note: Matplotlib automatically ignores the NaN values in 'B'
                plt.scatter(
                    foF2_and_fmin_present["UT+7"],
                    foF2_and_fmin_present["fmin"],
                    #label='fmin',
                    color=plot_markers_color,
                    marker='o'
                )

                # Plot 2: Column A vs C
                plt.scatter(
                    foF2_and_fmin_present["UT+7"],
                    foF2_and_fmin_present["foF2"],
                    #label="foF2",
                    color=plot_markers_color,
                    marker='o'
                )

                # Formatting the plot for a time axis
                #plt.xlabel('Time (Column C)')
                plt.xticks([])
                #plt.ylabel('Value (Columns A and B)')
                plt.yticks([])
                #plt.title(timeframe)
                plt.grid(axis='y', linestyle=':', alpha=0.6)

                # Get rid of borders
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                # Set background to black
                ax.set_facecolor(plot_bg_color)

                # Improve x-axis readability for datetime objects
                plt.gcf().autofmt_xdate()

                #plt.legend()
                plt.tight_layout()
                #plt.show()

                savepath = f"{main_directory}/{split_date_by}/{timeframe}_{day}.png"
                plt.savefig(savepath)
                plt.close()

def table_to_plot_pak_jiyo():
    main_directory = "/dataset/data_pak_jiyo/"
    filenames = os.listdir(main_directory)

table_to_plot_manual()
