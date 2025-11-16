import itertools
import os
import numpy as np
import pandas as pd
from typing import List

from eval_with_manual import get_metrics_single_numeric_with_manual

def get_metrics_all_months_individual(path_to_prepared_numeric_csv_dir: str, path_to_prepared_manual_dir: str, path_to_save_evaluation: str):
    """
    Getting RMSE, MSE, R coefficient for each month.
    """
    filenames = os.listdir(path_to_prepared_numeric_csv_dir)

    metrics_per_month = pd.DataFrame(columns=["RSE", "RMSE", "gmin", "gmax", "grange", "R"])

    for month in filenames:
        print(month)
        metrics_per_month.loc[len(metrics_per_month)] = get_metrics_single_numeric_with_manual(os.path.join(path_to_prepared_numeric_csv_dir, month), os.path.join(path_to_prepared_manual_dir, month))

    metrics_per_month.to_csv(path_to_save_evaluation)

def get_metrics_per_month_time(path_to_numeric_csv_dir: str, path_to_manual_dir: str, version: str):
    """
    Getting metrics per time slot, grouping all days of the month together.

    1. separate by month 
    2. for each day in month, group by time (group all fof2 and fmin)
    """
    numeric_filenames = os.listdir(path_to_numeric_csv_dir)

    # Sorting per month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    files_by_month = {col: [] for col in month_names}
    for name in numeric_filenames:
        splits = name.split("-")
        for month in month_names:   # Circumvents other files being processed (i.e. .DS_Store)
            if month in splits:
                files_by_month[month].append(name)

    # Getting time slots by float and analog for mask and column names respectively
    timeframes = np.arange(0, 24, 0.25)
    timeframes_analog = pd.date_range(
        start="00:00",
        end="23:45",
        freq="15min"
    ).strftime("%H:%M").tolist()

    final_RSE = {}
    final_RMSE = {} 
    final_MIN = {}
    final_MAX = {}
    final_RANGE = {}
    final_R = {}
    final_MBE = {}

    for month in files_by_month.keys():
        # Will store list of dicts for each day
        man_fmin_per_month = []
        man_fof2_per_month = []
        num_fmin_per_month = []
        num_fof2_per_month = []

        for day, filename in enumerate(files_by_month[month]):
            manual_df = pd.read_csv(os.path.join(path_to_manual_dir, filename))
            numeric_df = pd.read_csv(os.path.join(path_to_numeric_csv_dir, filename))
        
            man_fmin_per_day = {"day": day}
            man_fof2_per_day = {"day": day}
            num_fmin_per_day = {"day": day}
            num_fof2_per_day = {"day": day}

            # Get fmin and foF2 at each time
            for i in range(len(timeframes)-1):
                # Using right and left-inclusive intervals since some time slots might exclusively fall towards one boundary or the other
                # Should not be an issue to metrics as each time slot is independent of each other
                man_fmin = manual_df[(manual_df["time_as_float"] <= timeframes[i+1]) & (manual_df["time_as_float"] >= timeframes[i])]["fmin"].tolist()
                man_fof2 = manual_df[(manual_df["time_as_float"] <= timeframes[i+1]) & (manual_df["time_as_float"] >= timeframes[i])]["foES_foF2"].tolist()

                num_fmin = numeric_df[(numeric_df["time_as_float"] <= timeframes[i+1]) & (numeric_df["time_as_float"] >= timeframes[i])]["fmin"].tolist()
                num_fof2 = numeric_df[(numeric_df["time_as_float"] <= timeframes[i+1]) & (numeric_df["time_as_float"] >= timeframes[i])]["foF2"].tolist()

                # Might pull several values
                assert len(man_fmin) == len(num_fmin)
                assert len(man_fof2) == len(num_fof2)

                # Allowing multiple values for each time slot
                man_fmin_per_day[timeframes_analog[i+1]] = man_fmin
                man_fof2_per_day[timeframes_analog[i+1]] = man_fof2
                num_fmin_per_day[timeframes_analog[i+1]] = num_fmin
                num_fof2_per_day[timeframes_analog[i+1]] = num_fof2

            # After iterating through all time slots, 
            man_fmin_per_month.append(man_fmin_per_day)
            man_fof2_per_month.append(man_fof2_per_day)
            num_fmin_per_month.append(num_fmin_per_day)
            num_fof2_per_month.append(num_fof2_per_day)

        man_fmin_df = pd.DataFrame(man_fmin_per_month)
        man_fof2_df = pd.DataFrame(man_fof2_per_month)
        num_fmin_df = pd.DataFrame(num_fmin_per_month)
        num_fof2_df = pd.DataFrame(num_fof2_per_month)

        man_fmin_df.to_csv(f"./evaluations/{version}/per_time/man_fmin_df_per_time.csv")
        man_fof2_df.to_csv(f"./evaluations/{version}/per_time/man_fof2_df_per_time.csv")
        num_fmin_df.to_csv(f"./evaluations/{version}/per_time/num_fmin_df_per_time.csv")
        num_fof2_df.to_csv(f"./evaluations/{version}/per_time/num_fof2_df_per_time.csv")

        # Making sure that both DFs matchup
        assert len(man_fmin_df) + len(man_fof2_df) == len(num_fmin_df) + len(num_fof2_df)
        assert len(man_fmin_df) + len(num_fmin_df) == len(man_fof2_df) + len(man_fof2_df)

        # Storing RMSE by the month
        monthly_RSE = []
        monthly_RMSE = []
        monthly_MIN = []
        monthly_MAX = []
        monthly_RANGE = []
        monthly_R = []
        monthly_MBE = []

        # Getting metrics per time slot
        for time in timeframes_analog[1:]:
            # Converting to flats since some cells have multiple values
            man_fmin_flat = list(itertools.chain.from_iterable(man_fmin_df[time].tolist()))
            man_fof2_flat = list(itertools.chain.from_iterable(man_fof2_df[time].tolist()))
            num_fmin_flat = list(itertools.chain.from_iterable(num_fmin_df[time].tolist()))
            num_fof2_flat = list(itertools.chain.from_iterable(num_fof2_df[time].tolist()))

            total_values = len(man_fmin_flat) + len(man_fof2_flat)
            
            # If empty, metrics return as np.nan
            if total_values == 0:
                monthly_RSE.append(np.nan)
                monthly_RMSE.append(np.nan)
                monthly_MIN.append(np.nan)
                monthly_MAX.append(np.nan)
                monthly_RANGE.append(np.nan)
                monthly_R.append(np.nan)
                continue

            # Calculating RSE
            mean_manual = (sum(man_fmin_flat) + sum(man_fof2_flat)) / total_values

            top_rse, bot_rse = 0, 0
            top_rse = (
                (np.array(man_fmin_flat) - np.array(num_fmin_flat))**2 +
                (np.array(man_fof2_flat) - np.array(num_fof2_flat))**2
            ).sum()
            bot_rse = (
                (np.array(man_fmin_flat) - mean_manual)**2 +
                (np.array(man_fof2_flat) - mean_manual)**2
            ).sum()
            rse = top_rse / bot_rse

            # Calculating RMSE
            mse = (
                (np.array(man_fmin_flat) - np.array(num_fmin_flat))**2 +
                (np.array(man_fof2_flat) - np.array(num_fof2_flat))**2
            ).sum() / total_values
            rmse = np.sqrt(mse)

            gmin = min(num_fmin_flat)
            gmax = max(num_fof2_flat)
            grange = gmax-gmin

            correlation_matrix = np.corrcoef(man_fmin_flat + man_fof2_flat, num_fmin_flat + num_fof2_flat)
            r = correlation_matrix[0, 1]

            # Predicted - actual
            mbe = (
                (np.array(num_fmin_flat) - np.array(man_fmin_flat)) +
                (np.array(num_fof2_flat) - np.array(man_fof2_flat))
            ).sum() / total_values

            monthly_RSE.append(round(rse, 3))
            monthly_RMSE.append(round(rmse, 3))
            monthly_MIN.append(round(gmin, 3))
            monthly_MAX.append(round(gmax, 3))
            monthly_RANGE.append(round(grange, 3))
            monthly_R.append(round(r, 3))
            monthly_MBE.append(round(mbe, 3))

        # After getting all 96 time slots for the month, put into final dicts
        final_RSE[month] = monthly_RSE
        final_RMSE[month] = monthly_RMSE
        final_MAX[month] = monthly_MAX
        final_MIN[month] = monthly_MIN
        final_RANGE[month] = monthly_RANGE
        final_R[month] = monthly_R
        final_MBE[month] = monthly_MBE

    # Convert final metric dicts to csv and save
    print("Getting final CSVs")
    pd.DataFrame(final_RSE, index=timeframes_analog[1:]).to_csv(f"./evaluations/{version}/per_time/RSE_per_time.csv")
    pd.DataFrame(final_RMSE, index=timeframes_analog[1:]).to_csv(f"./evaluations/{version}/per_time/RMSE_per_time.csv")
    pd.DataFrame(final_MIN, index=timeframes_analog[1:]).to_csv(f"./evaluations/{version}/per_time/MIN_per_time.csv")
    pd.DataFrame(final_MAX, index=timeframes_analog[1:]).to_csv(f"./evaluations/{version}/per_time/MAX_per_time.csv")
    pd.DataFrame(final_RANGE, index=timeframes_analog[1:]).to_csv(f"./evaluations/{version}/per_time/RANGE_per_time.csv")
    pd.DataFrame(final_R, index=timeframes_analog[1:]).to_csv(f"./evaluations/{version}/per_time/R_per_time.csv")
    pd.DataFrame(final_MBE, index=timeframes_analog[1:]).to_csv(f"./evaluations/{version}/per_time/MBE_per_time.csv")
    print("Finished")

def get_metrics_per_season_day_night(path_to_numeric_csv_dir: str, path_to_manual_dir: str, version: str):
    """
    Grouping by seasons and day/night.
    
    Seasons:
        - Dec, Jan, Feb
        - Mar, Apr, May
        - Jun, Jul, Aug
        - Sep, Oct, Nov

    Daytime: 06:00 - 18:00
    Nightime: 18:00 - 06:00
    """
    numeric_filenames = os.listdir(path_to_numeric_csv_dir)

    # Sorting per month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    files_by_season = {"Dec_Jan_Feb": [], "Mar_Apr_May": [], "Jun_Jul_Aug": [], "Sep_Oct_Nov": []}
    for name in numeric_filenames:
        splits = name.split("-")
        for month in month_names:   # Circumvents other files being processed (i.e. .DS_Store)
            if month in splits:
                # Splitting by season
                labels = list(files_by_season.keys())
                if month in labels[0]:
                    files_by_season[labels[0]].append(name)
                if month in labels[1]:
                    files_by_season[labels[1]].append(name)
                if month in labels[2]:
                    files_by_season[labels[2]].append(name)
                if month in labels[3]:
                    files_by_season[labels[3]].append(name)

    # Getting time slots by float and analog for mask and column names respectively
    timeframes_labels = ["Daytime (06:00-18:00)", "Nighttime (18:00-06:00)"]

    final_RSE = {}
    final_RMSE = {} 
    final_MIN = {}
    final_MAX = {}
    final_RANGE = {}
    final_R = {}
    final_MBE = {}

    for month in files_by_season.keys():
        # Will store list of dicts for each day
        man_fmin_per_month = []
        man_fof2_per_month = []
        num_fmin_per_month = []
        num_fof2_per_month = []

        for day, filename in enumerate(files_by_season[month]):
            manual_df = pd.read_csv(os.path.join(path_to_manual_dir, filename))
            numeric_df = pd.read_csv(os.path.join(path_to_numeric_csv_dir, filename))
        
            man_fmin_per_day = {"day": day}
            man_fof2_per_day = {"day": day}
            num_fmin_per_day = {"day": day}
            num_fof2_per_day = {"day": day}

            # Getting data at the proper time slots
            # Only two timeframes to calculate (night/day)

            # Using right and left-inclusive intervals since some time slots might exclusively fall towards one boundary or the other
            # Should not be an issue to metrics as each time slot is independent of each other

            # Daytime
            man_fmin = manual_df[(manual_df["time_as_float"] <= 18) & (manual_df["time_as_float"] >= 6)]["fmin"].tolist()
            man_fof2 = manual_df[(manual_df["time_as_float"] <= 18) & (manual_df["time_as_float"] >= 6)]["foES_foF2"].tolist()

            num_fmin = numeric_df[(numeric_df["time_as_float"] <= 18) & (numeric_df["time_as_float"] >= 6)]["fmin"].tolist()
            num_fof2 = numeric_df[(numeric_df["time_as_float"] <= 18) & (numeric_df["time_as_float"] >= 6)]["foF2"].tolist()

            assert len(man_fmin) == len(num_fmin)
            assert len(man_fof2) == len(num_fof2)
            
            man_fmin_per_day[timeframes_labels[0]] = man_fmin
            man_fof2_per_day[timeframes_labels[0]] = man_fof2
            num_fmin_per_day[timeframes_labels[0]] = num_fmin
            num_fof2_per_day[timeframes_labels[0]] = num_fof2

            # Nighttime
            man_fmin = manual_df[(manual_df["time_as_float"] <= 6) | (manual_df["time_as_float"] >= 18)]["fmin"].tolist()
            man_fof2 = manual_df[(manual_df["time_as_float"] <= 6) | (manual_df["time_as_float"] >= 18)]["foES_foF2"].tolist()

            num_fmin = numeric_df[(numeric_df["time_as_float"] <= 6) | (numeric_df["time_as_float"] >= 18)]["fmin"].tolist()
            num_fof2 = numeric_df[(numeric_df["time_as_float"] <= 6) | (numeric_df["time_as_float"] >= 18)]["foF2"].tolist()

            assert len(man_fmin) == len(num_fmin)
            assert len(man_fof2) == len(num_fof2)

            man_fmin_per_day[timeframes_labels[1]] = man_fmin
            man_fof2_per_day[timeframes_labels[1]] = man_fof2
            num_fmin_per_day[timeframes_labels[1]] = num_fmin
            num_fof2_per_day[timeframes_labels[1]] = num_fof2

            # After iterating through all time slots, 
            man_fmin_per_month.append(man_fmin_per_day)
            man_fof2_per_month.append(man_fof2_per_day)
            num_fmin_per_month.append(num_fmin_per_day)
            num_fof2_per_month.append(num_fof2_per_day)

        man_fmin_df = pd.DataFrame(man_fmin_per_month)
        man_fof2_df = pd.DataFrame(man_fof2_per_month)
        num_fmin_df = pd.DataFrame(num_fmin_per_month)
        num_fof2_df = pd.DataFrame(num_fof2_per_month)

        man_fmin_df.to_csv(f"./evaluations/{version}/per_season/man_fmin_df_per_season.csv")
        man_fof2_df.to_csv(f"./evaluations/{version}/per_season/man_fof2_df_per_season.csv")
        num_fmin_df.to_csv(f"./evaluations/{version}/per_season/num_fmin_df_per_season.csv")
        num_fof2_df.to_csv(f"./evaluations/{version}/per_season/num_fof2_df_perseason.csv")

        # Making sure that both DFs matchup
        assert len(man_fmin_df) + len(man_fof2_df) == len(num_fmin_df) + len(num_fof2_df)
        assert len(man_fmin_df) + len(num_fmin_df) == len(man_fof2_df) + len(man_fof2_df)

        # Storing RMSE by the month
        monthly_RSE = []
        monthly_RMSE = []
        monthly_MIN = []
        monthly_MAX = []
        monthly_RANGE = []
        monthly_R = []
        monthly_MBE = []

        # Getting metrics per time slot
        for time in timeframes_labels:
            # Converting to flats since cells have multiple values
            man_fmin_flat = list(itertools.chain.from_iterable(man_fmin_df[time].tolist()))
            man_fof2_flat = list(itertools.chain.from_iterable(man_fof2_df[time].tolist()))
            num_fmin_flat = list(itertools.chain.from_iterable(num_fmin_df[time].tolist()))
            num_fof2_flat = list(itertools.chain.from_iterable(num_fof2_df[time].tolist()))

            total_values = len(man_fmin_flat) + len(man_fof2_flat)
            
            # If empty, metrics return as np.nan
            if total_values == 0:
                monthly_RSE.append(np.nan)
                monthly_RMSE.append(np.nan)
                monthly_MIN.append(np.nan)
                monthly_MAX.append(np.nan)
                monthly_RANGE.append(np.nan)
                monthly_R.append(np.nan)
                continue

            # Calculating RSE
            mean_manual = (sum(man_fmin_flat) + sum(man_fof2_flat)) / total_values

            top_rse, bot_rse = 0, 0
            top_rse = (
                (np.array(man_fmin_flat) - np.array(num_fmin_flat))**2 +
                (np.array(man_fof2_flat) - np.array(num_fof2_flat))**2
            ).sum()
            bot_rse = (
                (np.array(man_fmin_flat) - mean_manual)**2 +
                (np.array(man_fof2_flat) - mean_manual)**2
            ).sum()
            rse = top_rse / bot_rse

            # Calculating RMSE
            mse = (
                (np.array(man_fmin_flat) - np.array(num_fmin_flat))**2 +
                (np.array(man_fof2_flat) - np.array(num_fof2_flat))**2
            ).sum() / total_values
            rmse = np.sqrt(mse)

            gmin = min(num_fmin_flat)
            gmax = max(num_fof2_flat)
            grange = gmax-gmin

            correlation_matrix = np.corrcoef(man_fmin_flat + man_fof2_flat, num_fmin_flat + num_fof2_flat)
            r = correlation_matrix[0, 1]

            # Predicted - actual
            mbe = (
                (np.array(num_fmin_flat) - np.array(man_fmin_flat)) +
                (np.array(num_fof2_flat) - np.array(man_fof2_flat))
            ).sum() / total_values

            monthly_RSE.append(round(rse, 3))
            monthly_RMSE.append(round(rmse, 3))
            monthly_MIN.append(round(gmin, 3))
            monthly_MAX.append(round(gmax, 3))
            monthly_RANGE.append(round(grange, 3))
            monthly_R.append(round(r, 3))
            monthly_MBE.append(round(mbe, 3))

        # After getting all 96 time slots for the month, put into final dicts
        final_RSE[month] = monthly_RSE
        final_RMSE[month] = monthly_RMSE
        final_MAX[month] = monthly_MAX
        final_MIN[month] = monthly_MIN
        final_RANGE[month] = monthly_RANGE
        final_R[month] = monthly_R
        final_MBE[month] = monthly_MBE

    # Convert final metric dicts to csv and save
    print("Getting final CSVs")
    pd.DataFrame(final_RSE, index=timeframes_labels).to_csv(f"./evaluations/{version}/per_season/RSE_per_season.csv")
    pd.DataFrame(final_RMSE, index=timeframes_labels).to_csv(f"./evaluations/{version}/per_season/RMSE_per_season.csv")
    pd.DataFrame(final_MIN, index=timeframes_labels).to_csv(f"./evaluations/{version}/per_season/MIN_per_season.csv")
    pd.DataFrame(final_MAX, index=timeframes_labels).to_csv(f"./evaluations/{version}/per_season/MAX_per_season.csv")
    pd.DataFrame(final_RANGE, index=timeframes_labels).to_csv(f"./evaluations/{version}/per_season/RANGE_per_season.csv")
    pd.DataFrame(final_R, index=timeframes_labels).to_csv(f"./evaluations/{version}/per_season/R_per_season.csv")
    pd.DataFrame(final_MBE, index=timeframes_labels).to_csv(f"./evaluations/{version}/per_season/MBE_per_season.csv")
    print("Finished")

if __name__ == "__main__":
    version = "v28"

    # Getting metrics between model and manual, per month for every month
    #path_to_save_evaluation = f"./evaluations/{version}/all_months_individual_evals.csv"
    #get_metrics_all_months_individual_numeric_with_manual(path_to_save_prepared_numeric_csv_dir, path_to_save_prepared_manual_dir, path_to_save_evaluation) 
    
    path_to_save_prepared_numeric_csv_dir = f"./predictions/numeric_csv/prepared_for_numeric_eval/{version}"
    path_to_save_prepared_manual_dir = f"./dataset/data_scaling_manual/data_raw/prepared_for_numeric_eval/{version}"
    #get_metrics_per_month_time(path_to_save_prepared_numeric_csv_dir, path_to_save_prepared_manual_dir, version) 
    get_metrics_per_season_day_night(path_to_save_prepared_numeric_csv_dir, path_to_save_prepared_manual_dir, version) 
