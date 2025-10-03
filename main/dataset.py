import os
import numpy as np
import cv2 as cv
import torch
import torchvision.io as io
from pydantic import ValidationError
from torch.utils.data import Dataset

def preprocess_mask(mask):
    """ Extracting FTI class from mask. FTI class represented as 2.

    Args:
        - mask

    """
    return (mask == 2).astype(float)

## Dataloader
class FTIDataset(Dataset):
    """FIT Dataset; read images; apply augmentations.

    Args:
        images_directory (str): path to images
        masks_directory (str): path to masks
        class_values (list): list of classes
        transformation (albumentations.Compose): transformation pipeline

    """

    def __init__(self, images_directory, masks_directory, transformation=None):
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.all_files_in_images_directory = os.listdir(images_directory)
        self.all_files_in_masksk_directory = os.listdir(masks_directory)
        self.image_filenames = sorted([
            f for f in self.all_files_in_images_directory if f.endswith('.jpg')
        ])
        self.mask_filenames = sorted([
            f for f in self.all_files_in_masksk_directory if f.endswith('.png')
        ])
        assert len(self.image_filenames) == len(self.mask_filenames) # Ensuring folders have same number of files

        self.transformation = transformation

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]
        image_path = os.path.join(self.images_directory, image_name)
        mask_path = os.path.join(self.masks_directory, mask_name)

        # Reading and converting image/mask to RGB and grayscale
        image = cv.imread(image_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        assert image is not None and mask is not None   # Ensuring image and mask are read properly

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Extract classes from masks
        mask = preprocess_mask(mask)
        #print(image.shape, mask.shape) # (656, 875, 3) (656, 875)

        if self.transformation is not None:
            #transformed_image = self.transformation(image=image)
            #image = transformed_image["image"]
            #transformed_mask = self.transformation(image=mask)
            #mask = transformed_mask["image"]
            transformed = self.transformation(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        # Add extra dimension as channel to represent grayscale
        mask = np.expand_dims(mask, axis=-1)

        # Using float32 since MPS doesn't support float64
        image = torch.from_numpy(image).float()  # .float() casts to torch.float32
        mask = torch.from_numpy(mask).float()

        # Transpose images to match Pytorch format
        # Reshape to be (C, H, W)
        image = image.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        #print(image.shape, mask.shape) # (C, H, W)

        return image, mask
        
    def __len__(self):
        return(len(self.image_filenames))
    
    def get_name(self, idx):
        return self.image_filenames[idx]
