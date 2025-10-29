from collections import defaultdict
import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_min_max(path_to_mask: str):
    """

    """
    # TODO: very right part of mask is straight line
    mask = cv.imread(path_to_mask)
    mask_dims = mask.shape
    mask_width, mask_height = mask_dims[0], mask_dims[1]    # width is x, height is y
    assert mask is not None, "file could not be read, check with os.path.exists()"
    edges = cv.Canny(mask, 100, 200)
    #print(edges[:, 609]) # col view
    #print(edges[600, :]) # row view

    float_indices = np.linspace(0, mask_height, 96)
    sampled_col_indices = np.round(float_indices).astype(int)
    sampled_col_indices = sampled_col_indices[sampled_col_indices < mask_height]    # Ensure the indices do not exceed the array bounds (639)

    sampled_edges = edges[:, sampled_col_indices]
    row_indices, col_indices = np.where(sampled_edges > 0)

    full_range = np.arange(96)
    missing_time_indices = np.setdiff1d(full_range, col_indices)

    row_indices = 640 - row_indices
    col_indices = sampled_col_indices[col_indices]          # Map the 0-95 column indices back to the 0-639 original indices

    row_indices_scaled = row_indices * 20 / mask_height
    col_indices_scaled = col_indices * 24 / mask_height

    grouped_row_indices = defaultdict(list)
    for x_idx, y_idx in zip(col_indices_scaled, row_indices_scaled):
        grouped_row_indices[x_idx].append(y_idx)

    final_y_coords = {}

    for x_idx, y_list in grouped_row_indices.items():
        if len(y_list) >= 2:
            y_min = np.min(y_list)
            y_max = np.max(y_list)
            
            # Store the two required Y-values
            final_y_coords[x_idx] = [y_min, y_max]
        else:
            # Handle the unexpected case where an X-value has less than 2 Y-values.
            # You might skip it or assign a placeholder like [np.nan, np.nan]
            final_y_coords[x_idx] = [np.nan, np.nan]

    present_indices = list(final_y_coords.keys())
    y_min_values = [v[0] for v in final_y_coords.values()]
    y_max_values = [v[1] for v in final_y_coords.values()]
    for i in missing_time_indices:
        y_min_values.insert(i, np.nan)
        y_max_values.insert(i, np.nan)

    start_time = "00:00"
    end_time = "23:45"
    time_intervals = pd.date_range(start=start_time, end=end_time, freq="15min").time

    data_rows = {"Time": time_intervals, "fmin": y_min_values, "foF2": y_max_values}
    series = pd.DataFrame(data_rows)
    print(series.head())

    # TODO: FIX PLOT
    plot = True 
    if plot:
        X = series["Time"].apply(
    lambda t: t.hour + (t.minute / 60) + (t.second / 3600)
)
        Y1 = series["fmin"]
        Y2 = series["foF2"]

        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(mask, cmap = 'gray')
        plt.title('Original Image')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(132)
        plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(133)
        plt.plot(col_indices_scaled, row_indices_scaled, marker='o', linestyle='-', color='blue', label='Y1 Series')
        #plt.plot(X, Y1, marker='o', linestyle='-', color='blue', label='Y1 Series')
        #plt.plot(X, Y2, marker='x', linestyle='--', color='red', label='Y2 Series')

        plt.xlim(0, 24) 
        plt.ylim(0, 20)

        plt.tight_layout()
     
        plt.show()

version = "v16"
index = 4
mask_directory = f"./predictions/mask_only/{version}"
mask_filenames = os.listdir(mask_directory)

get_min_max(os.path.join(mask_directory, mask_filenames[index]))
