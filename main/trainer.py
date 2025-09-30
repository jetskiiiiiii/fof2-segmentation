import lightning as L
from model import FO2Model
from dataloader import train_loader, val_loader

# Training hyperparameters and settings
MAX_EPOCHS = 20

encoder_name = "resnet18" # Using encoder with smallest params
encoder_weights = "imagenet"
in_channels = 3
classes = 1

# Passing 3 architectures to train
model_unet = FO2Model(architecture="Unet", encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
model_deeplabv3 = FO2Model(architecture="deeplabv3", encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
model_fpn = FO2Model(architecture="fpn", encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)

for task in [model_unet, model_deeplabv3, model_fpn]:
    trainer = L.Trainer(max_epochs=MAX_EPOCHS, accelerator="mps", devices=1) # Running on Apple Silicon
    trainer.fit(task, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f"{task.architecture_name}_best.ckpt")
