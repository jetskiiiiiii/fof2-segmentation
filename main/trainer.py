import lightning as L
from model import FO2Model
from dataloader import train_loader, val_loader
from lightning.pytorch.loggers import CSVLogger

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

logger = CSVLogger("logs", name="training_log")

trainer = L.Trainer(max_epochs=MAX_EPOCHS, accelerator=device, devices=1, logger=logger) # Running on Apple Silicon
trainer.fit(model_unet, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.save_checkpoint(f"{model_unet.architecture_name}_best.ckpt")
