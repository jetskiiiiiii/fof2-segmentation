import os
import numpy as np
import lightning as L
import matplotlib.pyplot as plt
from model import FO2Model
from dataloader import test_loader

def get_predictions(version: str, only_save_mask: bool, index: int | None = None):
    model_path = f"./logs/training_log/{version}/checkpoints/{version}.ckpt"

    model = FO2Model.load_from_checkpoint(model_path)
    model.eval()
    model.freeze()

    # Using predict_step
    trainer = L.Trainer()
    predictions = trainer.predict(model, dataloaders=test_loader) # Returns list containing one Tensor of torch.Size([13, 1, 640, 640])

    dpi = 100
    fig_dim = 640 / dpi

    # If no specific image is chosen, plot all
    if index is None:
        for batch_idx, (images, masks, filenames) in enumerate(test_loader):
            for i in range(len(filenames)):
                mask = predictions[batch_idx][i]
                print(mask.shape)

                fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)

                ax.imshow(
                    mask,
                    cmap="binary_r",
                    alpha=0.4,
                    zorder=0
                )

                if only_save_mask:
                    ax.set_xticks([])
                    ax.set_yticks([])

                    plt.tight_layout()
                    #plt.show()
                    path_to_save = f"./predictions/mask_only/{version}"
                    os.makedirs(path_to_save, exist_ok=True)
                    plt.savefig(f"{path_to_save}/{filenames[i]}", format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.close()
                    continue


                single_image_tensor = images[i]
                permuted_image_tensor = single_image_tensor.permute(1, 2, 0)
                image = permuted_image_tensor.numpy().astype(np.uint8)

                ax.imshow(
                    image,
                    zorder=0
                )


                ax.set_xticks([])
                ax.set_yticks([])

                plt.tight_layout()
                #plt.show()
                path_to_save = f"./predictions/overlay/{version}"
                os.makedirs(path_to_save, exist_ok=True)
                plt.savefig(f"{path_to_save}/{filenames[i]}", format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
                plt.close()

        return None

    else:
        batch_idx = index // 8 - 1
        index = index % 8

        batch = next(iter(test_loader))
        filename = batch[2][index]

        mask = predictions[batch_idx][index]

        fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)
        ax.imshow(
            mask,
            cmap="binary_r",
            alpha=0.4,
            zorder=1
        )

        if only_save_mask:
            ax.set_xticks([])
            ax.set_yticks([])

            plt.tight_layout()
            #plt.show()
            path_to_save = f"./predictions/mask_only/{version}"
            os.makedirs(path_to_save, exist_ok=True)
            plt.savefig(f"{path_to_save}/{filename}", format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            return None

        image_tensor = batch[0]
        single_image_tensor = image_tensor[index]
        permuted_image_tensor = single_image_tensor.permute(1, 2, 0)
        image = permuted_image_tensor.numpy().astype(np.uint8)

        ax.imshow(
            image,
            zorder=0
        )

        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.show()

get_predictions("v17", False)
