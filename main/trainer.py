import lightning as L
from torch._prims_common import dtype_to_type
from model import FO2Model
from dataloader import train_loader, val_loader

# Training hyperparameters and settings
MAX_EPOCHS = 20

encoder_name = "resnet18" # Using encoder with smallest params
encoder_weights = "imagenet"
in_channels = 3
classes = 1
device = "mps"

# Passing 3 architectures to train
model_unet = FO2Model(architecture="Unet", encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, device=device)
model_deeplabv3 = FO2Model(architecture="deeplabv3", encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, device=device)
model_fpn = FO2Model(architecture="fpn", encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes, device=device)

for task in [model_unet, model_deeplabv3, model_fpn]:
    trainer = L.Trainer(max_epochs=MAX_EPOCHS, accelerator=device, devices=1) # Running on Apple Silicon
    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f"{task.architecture_name}_best.ckpt")
