import torch
from model import FO2Model
from visualize import visualize

unet_model_path = "Unet_best.ckpt"
test_directory = "./dataset/test/test_images/"

model = FO2Model.load_from_checkpoint(unet_model_path)
x = 
model.eval()
with torch.no_grad():
    y_hat = model(x)
    
    # See masks
    visualize()
