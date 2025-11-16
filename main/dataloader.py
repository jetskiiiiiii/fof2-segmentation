from torch.utils.data import DataLoader
from dataset import FTIDataset
from transformation import train_transformation, eval_transformation

# Directories
image_train_path = "./dataset/train_2019_only/train_2019_only_images"
mask_train_path = "./dataset/train_2019_only/train_2019_only_masks"

image_val_path = "./dataset/val_2019_only/val_2019_only_images"
mask_val_path = "./dataset/val_2019_only/val_2019_only_masks"

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

