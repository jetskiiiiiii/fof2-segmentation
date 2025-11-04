import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from visualize import overlay_any_mask_to_image


def get_numeric_as_csv(path_to_mask: str, path_to_save: str):
    """

    """
    mask = cv.imread(path_to_mask)
    assert mask is not None, "file could not be read, check with os.path.exists()"
    mask_dims = mask.shape
    mask_width, mask_height = mask_dims[0], mask_dims[1]    # width is x, height is y
    edges = cv.Canny(mask, 100, 200)
    #print(edges[:, 609]) # col view
    #print(edges[600, :]) # row view

    # To get per 15 minutes, we divide the plot by 96,
    # since the predicted mask has no knowledge of time.
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

    start_time = "00:00"
    end_time = "23:45"
    time_intervals = pd.date_range(start=start_time, end=end_time, freq="15min").time

    final_y_coords = {}
    for i, (x_idx, y_list) in enumerate(grouped_row_indices.items()):
        if len(y_list) > 1:
            y_min = np.min(y_list)
            y_max = np.max(y_list)
            final_y_coords[x_idx] = [y_min, y_max]
        else:
            # Handle the unexpected case where an X-value has less than 2 Y-values.
            final_y_coords[x_idx] = [np.nan, np.nan]
    # Sort dict
    final_y_coords = dict(sorted(final_y_coords.items(), key=lambda item: item[0]))

    y_min_values = [v[0] for v in final_y_coords.values()]
    y_max_values = [v[1] for v in final_y_coords.values()]
    x_values = list(final_y_coords.keys())
    for i in missing_time_indices:
        y_min_values.insert(i, np.nan)
        y_max_values.insert(i, np.nan)
        x_values.insert(i, np.nan)

    data_rows = pd.DataFrame({"time_as_float": x_values, "time": time_intervals, "fmin": y_min_values, "foF2": y_max_values})
    df = pd.DataFrame(data_rows)
    df.to_csv(path_to_save)

def plot_numeric(path_to_csv: str, path_to_save: str): 
    df = pd.read_csv(path_to_csv)

    #if mask:
    #    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    #    ax.imshow(mask, cmap = 'gray')
    #    ax.set_title('Original Image')
    #    ax.set_xticks([])
    #    ax.set_yticks([])

    #if edges:
    #    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    #    ax.imshow(edges,cmap = 'gray')
    #    ax.set_title('Edge Image')
    #    ax.set_xticks([])
    #    ax.set_yticks([])

    X = df["time_as_float"]
    Y1 = df["fmin"]
    Y2 = df["foF2"]

    dpi = 100
    fig_dim = 640 / dpi

    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.set_facecolor("black")
    ax.vlines(
        x=X,
        ymin=Y1,
        ymax=Y2,
        color="white",
        linestyle="-",
        alpha=1,
        linewidth=5,
        zorder=2
    )
    ax.scatter(
        X,
        Y1,
        color="white",
        s=10,
        alpha=1,
        zorder=2
    )
    ax.scatter(
        X,
        Y2,
        color="white",
        s=10,
        alpha=1,
        zorder=2
    )
    ax.set_xlim(0, 24) 
    ax.set_ylim(0, 20)

    ax.axis("off")

    plt.savefig(path_to_save, format='jpg', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    version = "v22"
    mask_directory = f"./predictions/mask/{version}"
    mask_filenames = os.listdir(mask_directory)

    index = 0
    #for index in range(index, index+1):
    for index in range(len(mask_filenames)):
        root, ex = os.path.splitext(mask_filenames[index])
        if root == ".DS_Store":
            continue
        path_to_mask = os.path.join(mask_directory, mask_filenames[index])
        path_to_save_csv = f"./predictions/numeric_csv/{version}/{root}.csv"
        path_to_save_plot = f"./predictions/numeric_plot/{version}/{root}.jpg"
        path_to_save_plot_overlay = f"./predictions/numeric_plot_overlay/{version}/{root}.jpg"

        path_to_test_images = f"./dataset/test/test_images/{mask_filenames[index]}"

        get_numeric_as_csv(path_to_mask, path_to_save_csv)
        plot_numeric(path_to_save_csv, path_to_save_plot)
        overlay_any_mask_to_image(path_to_save_plot, path_to_test_images, path_to_save_plot_overlay)

