import os
import numpy as np
import cv2 as cv
import torch
import torchvision.io as io
from pydantic import ValidationError
from torch.utils.data import Dataset

## Dataloader
class FTIDataset(Dataset):
    """FIT Dataset; read images; apply augmentations.

    Args:
        images_directory (str): path to images
        masks_directory (str): path to masks
        class_values (list): list of classes
        transformation (albumentations.Compose): transformation pipeline

    """

    def __init__(self, images_directory, masks_directory, class_name=None, transformation=None):
        self.ids = os.listdir(images_directory)
        #self.images_files = [os.path.join(images_directory, images_id) for images_id in self.ids]
        #self.masks_files = [os.path.join(masks_directory, images_id) for images_id in self.ids]

        self.images_dir = images_directory
        self.masks_dir = masks_directory
        self.images_filenames = sorted(os.listdir(self.images_dir))
        self.masks_filenames = sorted(os.listdir(self.masks_dir))


        self.class_name = class_name
        self.transformation = transformation

    def __getitem__(self, idx):
        #image = io.read_image(self.images_files[idx])
        #image = image.permute(1, 2, 0).contiguous().numpy()

        #image = cv.imread(self.images_files[idx])
        
        #mask = io.read_image(self.masks_files[idx])
        #mask = mask.permute(1, 2, 0).contiguous().numpy()

        #mask = cv.imread(self.masks_files[idx])

        img_name = self.images_filenames[idx]
        mask_name = self.masks_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)
        image = cv.imread(img_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        # Converting images to RGB and masks to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        #print(image.shape, mask.shape) 

        # Extract classes from masks
        masks = (mask == self.class_name)
        mask = np.stack(masks, axis=-1).astype("float")
        mask = mask.transpose(1, 0)
        #print(image.shape, mask.shape) # (656, 875, 3) (875, 4, 656)

        if self.transformation:
            #transformed_sample = self.transformation(image=image, mask=mask)
            #image, mask = transformed_sample["image"], transformed_sample["mask"]

            transformed_image = self.transformation(image=image)
            image = transformed_image["image"]
            transformed_mask = self.transformation(image=mask)
            mask = transformed_mask["image"]
        #print(image.shape, mask.shape) # (H, W, C)

        mask = np.expand_dims(mask, axis=-1)
        # Using float32 since MPS doesn't support float64
        image = torch.from_numpy(image).float()  # .float() casts to torch.float32
        mask = torch.from_numpy(mask).float()

        #print(image.shape, mask.shape) 
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        #print(image.shape, mask.shape) # (656, 875, 3) (875, 4, 656)

        # Transpose images to match Pytorch format
        return image, mask
        
    def __len__(self):
        return(len(self.ids))
