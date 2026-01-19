import re
import os
import torch 
import cv2 as cv
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import FormatStrFormatter

from dataset import FTIDataset, preprocess_mask
from transformation import train_transformation, eval_transformation

def plot_image(image_path: str, path_to_save: str):
    """

    Args:
        - image_path (str): Path to raw image (jpg)
    """
    img = mpimg.imread(image_path)
    assert img is not None

    img = train_transformation(image=img)
    img = img["image"]

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path_to_save, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

def plot_mask(mask_path):
    """

    Args:
        - mask_path (str): Path to string (png)
    """
    mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
    assert mask is not None
    mask = preprocess_mask(mask)
    fig, ax = plt.subplots()
    ax.imshow(mask, interpolation="nearest", cmap="binary_r")

    ax.axis('off')
    plt.show()

# Visualize data
def visualize_image_mask(image=None, mask=None):
    """PLot images in one row."""

    plt.figure(figsize=(10, 5))
    if image is not None:
        plt.subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title("Image")
        plt.imshow(image)

    if mask is not None:
        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask)

    plt.show()

def plot_max_edges_of_mask(path_to_mask: str, path_to_save: str): 
    """
    Reads an image, applies Canny edge detection, finds the maximum Y-coordinate 
    (row index) of an edge for every X-coordinate (column), and saves the plot.
    
    Args:
        path_to_mask (str): Path to the input image mask.
        path_to_save (str): Directory where the plot should be saved.
    """ 
    mask = cv.imread(path_to_mask, cv.IMREAD_GRAYSCALE) # Read as grayscale
    assert mask is not None, "file could not be read, check with os.path.exists()"
    
    image_height, image_width = mask.shape[0], mask.shape[1] 
    
    # Apply Canny edge detection
    edges = cv.Canny(mask, 100, 200)

    # --- Find Edge Coordinates and Max Y per Column ---
    # np.where returns (row_indices, col_indices), which corresponds to (y, x)
    row_indices, col_indices = np.where(edges > 0)
    max_y_per_x = {}
    
    # Iterate over all edge coordinates to find the max row index (max y) for each column (x)
    for x, y in zip(col_indices, row_indices):
        # Max y means the lowest point on the edge in the image coordinate system.
        if x not in max_y_per_x or y < max_y_per_x[x]:
            max_y_per_x[x] = y
            
    # Prepare the coordinates for plotting
    x_coords = sorted(max_y_per_x.keys())
    y_coords = [max_y_per_x[x] for x in x_coords]

    # --- Plot the Data using fig and ax and Save ---
    
    # 1. Create a figure (fig) and an axes (ax) object
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    
    ax.set_facecolor("black")
    
    # 2. Plot the data on the axes
    ax.plot(x_coords, y_coords, marker='o', linestyle='-', color='white', markersize=2, label='Max Edge Point')
    
    # 3. Set titles, labels, and limits using the ax object
    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Invert the y-axis to match the standard image convention (y=0 is the top)
    ax.invert_yaxis() 
    
    # 4. Define the save path and save the figure
    fig.savefig(path_to_save, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig) # Close the figure to free up memory

def display_image_grid(dataset):
    image_filenames = dataset.image_filenames
    images_directory = dataset.images_directory
    masks_directory = dataset.masks_directory

    cols = 2
    rows = len(image_filenames)
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(image_filenames):
        image = cv.imread(os.path.join(images_directory, image_filename), cv.IMREAD_COLOR_RGB)

        mask = cv.imread(os.path.join(masks_directory, image_filename.replace(".jpg", ".png")), cv.IMREAD_UNCHANGED)
        mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest", cmap="gray")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

    plt.tight_layout()
    plt.show()

### Code from https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/
### Will be implemented later

# Simple function to overlay mask on image for visualization
def overlay_mask(image, mask, alpha=0.5, color=(0, 1, 0)): # Green overlay
    if isinstance(mask, list):
        mask = np.array(mask)
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    color_array = (np.array(color) * 255).astype(np.uint8)
    # Create a color overlay where mask is > 0
    mask_overlay[mask > 0] = color_array

    # Blend image and overlay
    overlayed_image = cv.addWeighted(image, 1, mask_overlay, alpha, 0)
    return overlayed_image


def visualize_segmentation(dataset, idx=-1, samples=3):
    figure, ax = plt.subplots(samples + 1, 2, figsize=(8, 4 * (samples + 1)))

    # --- Get the original image and mask --- #
    original_transform = dataset.transformation
    dataset.transformation = None # Temporarily disable for raw data access
    image, mask, _ = dataset[idx]
    dataset.transformation = original_transform # Restore

    image = image.permute(1, 2, 0)
    mask = mask.permute(1, 2, 0)
    # Display original
    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(mask, cmap='gray') # Show mask directly
    ax[0, 1].set_title("Original Mask")
    ax[0, 1].axis("off")
    # ax[0, 1].imshow(overlay_mask(image, mask)) # Or show overlay
    # ax[0, 1].set_title("Original Overlay")

    # --- Apply and display augmented versions --- #
    for i in range(samples):
        aug_image, aug_mask = image, mask # Should not happen normally

        # Display augmented image and mask
        ax[i + 1, 0].imshow(aug_image)
        ax[i + 1, 0].set_title(f"Augmented Image {i+1}")
        ax[i + 1, 0].axis("off")

        ax[i + 1, 1].imshow(aug_mask, cmap='gray') # Show mask directly
        ax[i + 1, 1].set_title(f"Augmented Mask {i+1}")
        ax[i + 1, 1].axis("off")
        # ax[i+1, 1].imshow(overlay_mask(aug_image, aug_mask)) # Or show overlay
        # ax[i+1, 1].set_title(f"Augmented Overlay {i+1}")


    plt.tight_layout()
    plt.show()

# Assuming train_dataset is created with train_transform:
# visualize_segmentation(train_dataset, samples=3)

def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
    cols = 3 if predicted_masks else 2
    rows = len(images_filenames)
    _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(images_filenames):
        image = cv.imread(os.path.join(images_directory, image_filename), cv.IMREAD_COLOR_RGB)

        mask = cv.imread(os.path.join(masks_directory, image_filename.replace(".jpg", ".png")), cv.IMREAD_UNCHANGED)
        mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()

def overlay_any_mask_to_image(path_to_mask: str, path_to_image: str, path_to_save: str, is_image_tensor: bool =False):
    image = None
    if is_image_tensor:
        image_tensor = torch.load(path_to_image, weights_only=False)
        permuted_image_tensor = image_tensor.permute(1, 2, 0)
        image = permuted_image_tensor.numpy().astype(np.uint8)
    else:
        image = cv.imread(path_to_image) 
        assert image is not None, "File could not be read."
        image = eval_transformation(image=image)
        image = image["image"]
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # For compatibility with Matplotlib

    mask = mpimg.imread(path_to_mask)

    dpi = 100
    fig_dim = 640 / dpi

    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(
        mask,
        cmap="binary_r",
        alpha=0.4,
        zorder=1,
    )

    ax.imshow(
        image,
        zorder=0
    )

    ax.axis("off")

    plt.savefig(path_to_save, format='jpg', pad_inches=0)
    plt.close()

def overlay_fof2_from_csv(path_to_image: str, path_to_man_csv: str, path_to_save_overlay: str, is_manual: bool, fo_type: str, title: str, path_to_qs_csv: str = None, path_to_num_csv: str = None) -> None:
    man_table = pd.read_csv(path_to_man_csv)

    if fo_type == "foES_foF2":
        man_table["foES_foF2"] = np.where(man_table["foF2"].notna(), man_table["foF2"], man_table["foES"])

    man_plot_data = {
        "man_x": man_table["time_as_float"],
        "man_y": man_table["foF2"] if fo_type == "foF2" else man_table["foES_foF2"],
    }

    if path_to_qs_csv:
        qs_table = pd.read_csv(path_to_qs_csv)
        qs_filtered_table = qs_table[qs_table['Parameter'] == 'foF2']
        qs_plot_data = {
            "qs_x": qs_filtered_table["JamDec"],
            "qs_y": qs_filtered_table["Nilai"]
        }

    if path_to_num_csv:
        num_table = pd.read_csv(path_to_num_csv)
        num_plot_data = {
            "num_x": num_table["time_as_float"],
            "num_y": num_table["foF2"]
        }

    # 2. Setup Figure and Axes
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    FONT_SIZE_AXIS = 16
    FONT_SIZE_TITLE = 18
    FONT_SIZE_LEGEND = 12

    # 3. Load Image
    img = mpimg.imread(path_to_image)

    assert img is not None

    img = eval_transformation(image=img)
    img = img["image"]
    img = np.flipud(img)

    # 4. Define and Apply Extent for Image Alignment
    # Assuming typical plot limits for time (0-24 h) and frequency (0-15 MHz)
    x_min, x_max = 0.0, 24.0
    y_min, y_max = 0.0, 20.0 
    
    # Display the image as background
    # 'origin='lower'' ensures the image aligns with the standard plot origin (bottom-left)
    ax.imshow(img, aspect='auto', extent=[x_min, x_max, y_min, y_max], origin='lower', zorder=0, alpha=0.7)

    # 5. Plot the data points
    ax.scatter(
        man_plot_data["man_x"],
        man_plot_data["man_y"],
        s=25,          # Marker size
        color='lime',  # Marker color
        zorder=2,       # Ensure points are plotted above the background image
        label="Ground truth"
    )

    if path_to_qs_csv:
        ax.scatter(
            qs_plot_data["qs_x"],
            qs_plot_data["qs_y"],
            s=25,          # Marker size
            color='blue',  # Marker color
            zorder=2,       # Ensure points are plotted above the background image
            label="QS"
        )

    if path_to_num_csv:
        ax.scatter(
            num_plot_data["num_x"],
            num_plot_data["num_y"],
            s=25,          # Marker size
            color='red',  # Marker color
            zorder=2,       # Ensure points are plotted above the background image
            label="FPN model"
        )


    # 6. Set Plot Properties
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel("Time", fontsize=FONT_SIZE_AXIS)
    ax.set_ylabel("Frequency (MHz)", fontsize=FONT_SIZE_AXIS)
    ax.set_xticks(np.arange(0, 24.1, 4))
    ax.set_yticks(np.arange(0, 20.1, 4))
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_AXIS)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=FONT_SIZE_LEGEND)

    # 7. Save the figure
    try:
        plt.tight_layout()
        plt.savefig(path_to_save_overlay, dpi=100, bbox_inches="tight", pad_inches=0)
        print(f"Overlay plot successfully saved to {path_to_save_overlay}")
    except Exception as e:
        print(f"Error saving file to {path_to_save_overlay}: {e}")
    finally:
        plt.close(fig)

def plot_per_time_graph(path_to_metric_csv: str, path_to_save_graph: str, metric_type: str):
    df = pd.read_csv(path_to_metric_csv)

    # 1. Extract Z values (95 rows x 12 columns: Jan to Dec)
    Z = df.iloc[:, 1:].values
    M, N = Z.shape # M=95, N=12

    # 2. Define coordinates for cell edges
    x_edges = np.arange(N + 1)
    y_edges = np.arange(M + 1)

    # Create the meshgrid (using default 'xy' indexing)
    X, Y = np.meshgrid(x_edges, y_edges)

    # 3. Axis Tick Labels
    month_labels = df.columns[1:].tolist() # Jan, Feb, ..., Dec
    time_labels = df.iloc[:, 0].tolist() # 00:15, 00:30, ...

    # Use only a subset of time labels to prevent clutter on the Y-axis
    y_tick_indices = np.arange(0, M, 8)
    y_tick_labels = [time_labels[i] for i in y_tick_indices]

    # 4. Create the plot
    fig = plt.figure(figsize=(24, 18))
    ax1 = fig.add_subplot(111)
    FONT_SIZE = 35

    # Use pcolormesh with the extracted Z values and edge coordinates
    original_cmap = plt.get_cmap("viridis") if metric_type == "RMSE" else plt.get_cmap("hot")
    # Make a copy to avoid modifying the global 'plasma' colormap
    custom_cmap = original_cmap.copy()
    # Set the color for values below vmin to white
    custom_cmap.set_under('white')
    # Set the color for values above vmax to white
    custom_cmap.set_over('white')
    # Set the color for NaN/masked values to white
    custom_cmap.set_bad('white')
    if metric_type == "MBE":
        mesh = ax1.pcolormesh(X, Y, Z, edgecolors='w', linewidth=0.1, cmap=custom_cmap, vmin=-1.0, vmax=1.5)
    else:
        mesh = ax1.pcolormesh(X, Y, Z, edgecolors='w', linewidth=0.1, cmap=custom_cmap, vmin=0, vmax=2)

    # Set title and labels
    ax1.set_title(f"{metric_type} Intensity Plot", fontsize=FONT_SIZE)
    ax1.set_xlabel("Month", fontsize=FONT_SIZE)
    ax1.set_ylabel("Time", fontsize=FONT_SIZE)

    # Set X-axis ticks (centered on cells) and labels (Months)
    ax1.set_xticks(np.arange(N) + 0.5)
    ax1.set_xticklabels(month_labels, rotation=45, ha="right")

    # Set Y-axis ticks (centered on cells) and labels (Time)
    ax1.set_yticks(y_tick_indices + 0.5)
    ax1.set_yticklabels(y_tick_labels)
    ax1.set_ylim(0, M) # Invert y-axis to have 00:15 at the top

    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax1)
    cbar.set_label(f"{metric_type} Value (Z)", rotation=270, labelpad=30, fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.tight_layout()
    plt.savefig(path_to_save_graph)

if __name__ == "__main__":
    DATA_DIRECTORY = "dataset/"
    #sample_image_path = os.path.join(DATA_DIRECTORY, "FTIF_LTPMP-14-Feb-2019.png")
    #image = mpimg.imread(sample_image_path)
    #transformed_image = train_transformation(image=image)['image']
    #visualize(image=transformed_image)

    #image_test_path = "./dataset/test/test_images"
    #mask_test_path = "./dataset/test/test_masks"
    #test_data = FTIDataset(
    #    image_test_path,
    #    mask_test_path,
    #    transformation=eval_transformation
    #)
    #image_names = os.listdir(image_test_path)

    #visualize_segmentation(test_data)

    # Works
    #plot_image("./dataset/test/test_images/FTIF_LTPMP-1-Dec-2020.jpg")
    #plot_mask("./dataset/test/test_masks/FTIF_LTPMP-7-Jan-2019.png")

    #overlay("./predictions/numeric_plot/v20/FTIF_LTPMP-1-Apr-2019.jpg", "./dataset/test/test_images/FTIF_LTPMP-1-Apr-2019.jpg", "./predictions/for_testing/FTIF_LTPMP-1-Apr-2019_numeric_overlay.jpg")

    version = "v29"
    erase_type = "erase_when_either"
    fo_type = "foF2"

    filename = "FTIF_LTPMP-1-Apr-2020"
    path_to_image = f"./dataset/2020_all/2020_all_images/{filename}.jpg"
    #path_to_save = f"./dataset/sample_transformations/{filename}"
    #plot_image(path_to_image, path_to_save)
    path_to_mask = f"./predictions/mask/{version}/{filename}.jpg"
    path_to_save = f"./predictions/max_edges/{version}/{filename}"
    #plot_max_edges_of_mask(path_to_mask, path_to_save)
    
    # Overlaying manual to image
    filename = "FTIF_LTPMP-1-Jul-2020"
    path_to_image = f"./dataset/2020_all/2020_all_images/{filename}.jpg"
    path_to_qs_csv = f"./dataset/data_pak_jiyo/1-Jul-2020.csv"
    path_to_manual_csv = f"./dataset/data_scaling_manual/data_raw/prepared_for_numeric_eval/{version}/{erase_type}/{filename}.csv"
    path_to_num_csv = f"./predictions/numeric_csv/prepared_for_numeric_eval/{version}/{erase_type}/{filename}.csv"
    path_to_save = f"./dataset/data_pak_jiyo/overlay/{fo_type}/{filename}.jpg"
    #overlay_fof2_from_csv(path_to_image, path_to_manual_csv, path_to_save, True, fo_type, path_to_qs_csv, path_to_num_csv)
    # Multiple
    dir = f"./dataset/2020_all/2020_all_images/"
    files = os.listdir(dir)
    #for file in files:
    #    filename, _ = os.path.splitext(file)
    #    split_filename = re.split(r'[_\\-]', filename)
    #    alt_filename = f"{split_filename[2]}-{split_filename[3]}-{split_filename[4]}"
    #    path_to_image = f"./dataset/2020_all/2020_all_images/{filename}.jpg"
    #    path_to_manual_csv = f"./dataset/data_scaling_manual/data_raw/prepared_for_numeric_eval/{version}/{erase_type}/{filename}.csv"
    #    path_to_qs_csv = f"./dataset/data_pak_jiyo/{alt_filename}.csv"
    #    path_to_num_csv = f"./predictions/numeric_csv/prepared_for_numeric_eval/{version}/{erase_type}/{filename}.csv"
    #    path_to_save = f"./dataset/data_scaling_manual/data_raw/overlay/foF2/{filename}.jpg"
    #    overlay_fof2_from_csv(path_to_image, path_to_manual_csv, path_to_save, True, fo_type, alt_filename)
        #overlay_fof2_from_csv(path_to_image, path_to_manual_csv, path_to_save, True, fo_type, alt_filename, path_to_qs_csv, path_to_num_csv)

    metric_type = "RMSE"
    path_to_metric_csv = f"evaluations/{version}/{erase_type}/per_time/fof2_{metric_type}_per_time.csv"
    path_to_save_graph = f"evaluations/{version}/{erase_type}/per_time/fof2_{metric_type}_per_time.png"
    plot_per_time_graph(path_to_metric_csv, path_to_save_graph, metric_type)

    metric_type = "MBE"
    path_to_metric_csv = f"evaluations/{version}/{erase_type}/per_time/fof2_{metric_type}_per_time.csv"
    path_to_save_graph = f"evaluations/{version}/{erase_type}/per_time/fof2_{metric_type}_per_time.png"
    plot_per_time_graph(path_to_metric_csv, path_to_save_graph, metric_type)

