import os
import torch
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model import FO2Model
from dataloader import test_loader

def get_prediction_tensor(path_to_model: str, path_to_save: str):
    """
    Only performs inference on test data and saves predictions as tensor.
    """

    model = FO2Model.load_from_checkpoint(path_to_model)
    model.eval()
    model.freeze()

    # Using predict_step
    trainer = L.Trainer()
    predictions = trainer.predict(model, dataloaders=test_loader) # Returns list containing one Tensor of torch.Size([13, 1, 640, 640])

    torch.save(predictions, path_to_save)

def convert_all_predictions_to_mask_and_overlay(path_to_prediction_tensor: str, path_to_save_mask: str, path_to_save_overlay: str):
    """
    Converts entire prediction tensor to masks and overlays.
    """
    predictions = torch.load(path_to_prediction_tensor, weights_only=False)

    dpi = 100
    fig_dim = 640 / dpi

    for batch_idx, (images, masks, filenames) in enumerate(test_loader):
        for i in range(len(filenames)):
            batch = predictions[batch_idx]
            mask = batch[i] if batch.ndim == 3 else batch

            fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            ax.imshow(
                mask,
                cmap="binary_r",
                alpha=1,
                zorder=0
            )

            ax.axis("off")

            plt.savefig(f"{path_to_save_mask}/{filenames[i]}", format='jpg', pad_inches=0)
            plt.close(fig)


            single_image_tensor = images[i]
            permuted_image_tensor = single_image_tensor.permute(1, 2, 0)
            image = permuted_image_tensor.numpy().astype(np.uint8)

            fig_overlay, ax_overlay = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            ax_overlay.imshow(
                mask,
                cmap="binary_r",
                alpha=0.4, # Make it partially transparent
                zorder=1   # Ensure mask is on top of the image
            )

            ax_overlay.imshow(
                image,
                zorder=0
            )

            ax_overlay.axis("off")

            plt.savefig(f"{path_to_save_overlay}/{filenames[i]}", format='jpg', pad_inches=0)
            plt.close(fig_overlay)

def convert_single_prediction_to_mask(path_to_prediction_tensor: str, path_to_save: str, index: int, num_batch: int):
    """
    Converts a chosen prediction to a mask.
    """
    predictions = torch.load(path_to_prediction_tensor, weights_only=False)
    batch_idx = index // num_batch - 1
    index = index % num_batch 

    batch = next(iter(test_loader))
    filename = batch[2][index]

    mask = predictions[batch_idx][index]

    dpi = 100
    fig_dim = 640 / dpi

    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)

    ax.imshow(
        mask,
        cmap="binary_r",
        alpha=0.4,
        zorder=1
    )

    ax.set_xticks([])
    ax.set_yticks([])

    os.makedirs(path_to_save, exist_ok=True)
    plt.savefig(f"{path_to_save}/{filename}", format='jpg', pad_inches=0)
    plt.close()

def overlay_single_mask_with_image(path_to_mask: str, path_to_save: str, index: str):
    """
    Overlays mask with original image.
    """
    batch = next(iter(test_loader))
    image_tensor = batch[0]
    single_image_tensor = image_tensor[index]
    permuted_image_tensor = single_image_tensor.permute(1, 2, 0)
    image = permuted_image_tensor.numpy().astype(np.uint8)

    mask = mpimg.imread(path_to_mask)

    dpi = 100
    fig_dim = 640 / dpi

    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(
        mask,
        cmap="binary_r",
        alpha=0.4,
        zorder=1
    )

    ax.imshow(
        image,
        zorder=0
    )

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(path_to_save, format='jpg', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    version = "v22"
    model_path = f"./logs/training_log/{version}/checkpoints/{version}.ckpt"

    path_to_save_prediction_tensor = f"./predictions/prediction_tensor/{version}/{version}_prediction_tensor.pt"
    get_prediction_tensor(model_path, path_to_save_prediction_tensor)

    path_to_save_mask = f"./predictions/mask/{version}"
    path_to_save_overlay = f"./predictions/overlay/{version}"
    convert_all_predictions_to_mask_and_overlay(path_to_save_prediction_tensor, path_to_save_mask, path_to_save_overlay)
