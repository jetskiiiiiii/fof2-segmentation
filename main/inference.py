import os
import torch
import numpy as np
import lightning as L
from model import FO2Model
from visualize import visualize, overlay_mask
from dataloader import test_loader

unet_model_path = "Unet_best_1.ckpt"
test_images_directory = "./dataset/test/test_images/"
test_masks_directory = "./dataset/test/test_masks/"
image_filenames = os.listdir(test_images_directory)
mask_filenames = os.listdir(test_masks_directory)

# Hyperparameters that need to be passed because we forgot to call save_hyperparameters()
encoder_name = "resnet18" # Using encoder with smallest params
encoder_weights = "imagenet"
in_channels = 3
classes = 1
device = "mps"

model = FO2Model.load_from_checkpoint(unet_model_path, architecture="Unet", encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, device=device)
idx = 0
x = os.path.join(test_images_directory, image_filenames[idx])

# Using predict_step
trainer = L.Trainer()
predictions = trainer.predict(model, dataloaders=test_loader)[0] # Returns list containing one Tensor of torch.Size([13, 1, 640, 640])

# Using eval
#model.eval()
#with torch.no_grad():
#    batch = test_loader
#    pred = model(batch)
#
    # See masks
pred_tensor = predictions[idx].detach().cpu()
print(pred_tensor)
pred = (pred_tensor > 0).numpy().astype(np.uint8)
#print(pred[0, 0, :])
# print(pred.shape) Should be H, w

overlayed_image = overlay_mask(x, pred[0, 0, :])
visualize(overlayed_image)
