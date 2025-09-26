import os
import cv2 as cv
from torch.utils.data import Dataset

## Dataloader
class FITDataset(Dataset):
    """FIT Dataset; read images; apply augmentations.

    Args:
        images_directory (str): path to images
        masks_directory (str): path to masks
        class_values (list): list of classes
        transformation (albumentations.Compose): transformation pipeline

    """

    CLASSES = ["FTI"] # Unsure of what the classes are

    def __init__(self, images_directory, masks_directory, class_values=None, transformation=None):
        self.ids = os.listdir(images_directory)
        self.images_files = [os.path.join(images_directory, images_id) for images_id in self.ids]
        self.masks_files = [os.path.join(masks_directory, images_id) for images_id in self.ids]

        self.transformation = transformation

    def __getitem__(self, idx):
        # TODO: Might have to convert to RGB or grayscale
        image = cv.imread(self.images_files[idx])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        mask = cv.imread(self.masks_files[idx])
        mask = cv.cvtColor(mask, cv.IMREAD_GREYSCALE)

        if self.transformation:
            transformed_sample = self.transformation(image=image, mask=mask)
            image, mask = transformed_sample["image"], transformed_sample["mask"]

        # Transpose images to match Pytorch format
        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)
        
    def __len__(self):
        return(len(self.ids))
