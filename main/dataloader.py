import os
from pandas.core.common import random_state
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import FTIDataset, FTIDataset_No_Split
from transformation import train_transformation, eval_transformation, train_transformation_no_split, eval_transformation_no_split

def data_pre_split():
    # Directories
    image_train_path = "./dataset/2019/train/train_images"
    mask_train_path = "./dataset/2019/train/train_masks"

    image_val_path = "./dataset/2019/val/val_images"
    mask_val_path = "./dataset/2019/val/val_masks"

    image_test_path = "./dataset/2020_all/2020_all_images"
    mask_test_path = "./dataset/2020_all/2020_all_masks"

    # datasets
    train_data = FTIDataset(
        image_train_path,
        mask_train_path,
        transformation=train_transformation
    )

    val_data = FTIDataset(
        image_val_path,
        mask_val_path,
        transformation=eval_transformation
    )

    test_data = FTIDataset(
        image_test_path,
        mask_test_path,
        transformation=eval_transformation
    )

    # hyperparameters
    batch_size = 8

    # dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

def data_no_split(batch_size):
    # 2019 data
    images_path = "./dataset/2019/images"
    masks_path = "./dataset/2019/masks"

    image_test_path = "./dataset/2020_all/2020_all_images"
    mask_test_path = "./dataset/2020_all/2020_all_masks"

    all_filenames = sorted([
        f for f in os.listdir(images_path)
        if f.lower().endswith((".png", ".jpg")) and not f.startswith('.')   # Avoids .DS_Store
    ])

    train_files, val_files = train_test_split(
        all_filenames, test_size=0.15, random_state=42
    )

    train_data = FTIDataset_No_Split(images_path, masks_path, train_files, transformation=train_transformation_no_split)
    val_data = FTIDataset_No_Split(images_path, masks_path, val_files, transformation=eval_transformation_no_split)
    test_data = FTIDataset(
        image_test_path,
        mask_test_path,
        transformation=eval_transformation
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_no_split(8)
