import csv
import torch
import numpy as np
import gradio as gr
from PIL import Image

from get_numeric import get_numeric_as_csv
from model import FO2Model
from transformation import eval_transformation

DEVICE = "cpu"

version = "v28"
model_path = f"./logs/training_log/{version}/checkpoints/{version}.ckpt"
model = FO2Model.load_from_checkpoint(model_path)
model.to(DEVICE)
model.eval()

def get_segment(input_image_pil: Image.Image):
    # Pre-processing image to tensor
    image_np = np.array(input_image_pil)
    image = eval_transformation(image=image_np)
    image = image["image"]
    image_for_processing = image
    input_tensor = torch.from_numpy(image).float()
    input_tensor = input_tensor.permute(2, 0, 1)
    input_batch = input_tensor.unsqueeze(0)

    # Running inference directly through model's forward pass
    output_tensor = model(input_batch) 

    # Post-processing tensor back to image
    mask_tensor = torch.sigmoid(output_tensor).squeeze(0).squeeze(0)
    mask_bool = (mask_tensor > 0.5).numpy()

    overlay_image_np = image_for_processing.astype(np.float32) / 255.0
    OVERLAY_COLOR = np.array([1.0, 1.0, 1.0])
    ALPHA = 0.8
    
    colored_mask = np.zeros(overlay_image_np.shape, dtype=overlay_image_np.dtype)
    colored_mask[mask_bool] = OVERLAY_COLOR

    final_image = overlay_image_np.copy()

    final_image[mask_bool] = (
        ALPHA * colored_mask[mask_bool] +
        (1.0 - ALPHA) * overlay_image_np[mask_bool]
    )
    final_image = (final_image*255).clip(0, 255).astype(np.uint8)
    final_image_pil = Image.fromarray(final_image)

    # Getting CSV
    mask_filename = "./temp_app_files/mask.jpg"
    mask_int_8bit = (mask_bool * 255).astype(np.uint8)
    mask_image_pil = Image.fromarray(mask_int_8bit)
    mask_image_pil.save(mask_filename)
    
    numeric_csv_filename = "./temp_app_files/fmin_fof2_numeric.csv"
    get_numeric_as_csv(mask_filename, numeric_csv_filename)
    
    return final_image_pil, numeric_csv_filename

input_image = gr.Image(
    label="Input image",
    type="pil"
)
output_image = gr.Image(
    label="Output image",
    type="pil"
)
output_csv = gr.File(
    label="Download fmin, foF2 (CSV)",
    file_count="single"
)
gr.Interface(
    fn=get_segment,
    inputs=input_image,
    outputs=[output_image, output_csv],
    title="Critical frequency rapid determination",
).launch()
