import os
import lightning as L
import matplotlib.pyplot as plt
from model import FO2Model
from dataloader import test_loader, test_data

unet_model_path = "Unet_best_11.ckpt"
predictions_directory = "predictions/v11/"

model = FO2Model.load_from_checkpoint(unet_model_path)
model.eval()
model.freeze()

# Using predict_step
trainer = L.Trainer()
predictions = trainer.predict(model, dataloaders=test_loader) # Returns list containing one Tensor of torch.Size([13, 1, 640, 640])
predictions = predictions[0]

def save_prediction_as_jpg(predictions, predictions_directory):
    for i in range(len(predictions)):
        filename = os.path.join(predictions_directory, f"{test_data.get_name(i)}")
        os.makedirs(predictions_directory, exist_ok=True)

        fig, axes = plt.subplots(1, figsize=(10, 5))

        # Use 'binary' colormap for clear visualization of 0s and 1s
        # Invert the colormap if the background is 0 and you want it dark
        plt.imshow(predictions[i], cmap='binary_r') # 
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(filename, format='jpg', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

def plot_predictions(mask):
    fig, axes = plt.subplots(1, figsize=(10, 5))

    # Use 'binary' colormap for clear visualization of 0s and 1s
    # Invert the colormap if the background is 0 and you want it dark
    plt.imshow(mask, cmap='binary_r') # 
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    save_prediction_as_jpg(predictions, predictions_directory)
