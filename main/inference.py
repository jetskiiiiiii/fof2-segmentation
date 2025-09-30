import torch
from model import FO2Model
from visualize import visualize

model_path = ""

model = FO2Model.load_from_checkpoint(model_path)
x = 
model.eval()
with torch.no_grad():
    y_hat = model(x)
    
    # See masks
    visualize()
