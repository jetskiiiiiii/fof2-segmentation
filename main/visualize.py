import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataset import FTIDataset, preprocess_mask
from transformation import train_transformation, eval_transformation

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
    # Convert mask to 3 channels if needed, ensure boolean type
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    # Create a color overlay where mask is > 0
    mask_overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)

    # Blend image and overlay
    overlayed_image = cv.addWeighted(image, 1, mask_overlay, alpha, 0)
    return overlayed_image


def visualize_segmentation(dataset, idx=0, samples=3):
    # Make a copy of the transform list to modify for visualization
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        vis_transform = A.Compose(vis_transform_list)
    else:
        print("Warning: Could not automatically strip Normalize/ToTensor for visualization.")
        vis_transform = dataset.transform

    figure, ax = plt.subplots(samples + 1, 2, figsize=(8, 4 * (samples + 1)))

    # --- Get the original image and mask --- #
    original_transform = dataset.transform
    dataset.transform = None # Temporarily disable for raw data access
    image, mask = dataset[idx]
    dataset.transform = original_transform # Restore

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
        # Apply the visualization transform
        if vis_transform:
            augmented = vis_transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
        else:
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

if __name__ == "__main__":
    DATA_DIRECTORY = "dataset/"
    sample_image_path = os.path.join(DATA_DIRECTORY, "FTIF_LTPMP-14-Feb-2019.png")
    image = mpimg.imread(sample_image_path)
    transformed_image = train_transformation(image=image)['image']
    #visualize(image=transformed_image)

    image_test_path = "./dataset/test/test_images"
    mask_test_path = "./dataset/test/test_masks"
    test_data = FTIDataset(
        image_test_path,
        mask_test_path,
        transformation=eval_transformation
    )

    display_image_grid(test_data)
