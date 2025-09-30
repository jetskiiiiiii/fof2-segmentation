from torch.utils.data import DataLoader
from dataset import FITDataset
from transformation import train_transformation, eval_transformation

# Directories
image_train_path = "/Users/tprimandaru/Downloads/FoF2-v1/train/train_images"
mask_train_path = "/Users/tprimandaru/Downloads/FoF2-v1/train/train_images"

image_val_path = "/Users/tprimandaru/Downloads/FoF2-v1/val/val_images"
mask_val_path = "/Users/tprimandaru/Downloads/FoF2-v1/val/val_images"

image_test_path = "/Users/tprimandaru/Downloads/FoF2-v1/test/test_images"
mask_test_path = "/Users/tprimandaru/Downloads/FoF2-v1/test/test_images"

CLASSES = "FTI" # From roboflow

# Datasets
train_data = FITDataset(
    image_train_path,
    mask_train_path,
    class_name=CLASSES,
    transformation=train_transformation
)

val_data = FITDataset(
    image_val_path,
    mask_val_path,
    class_name=CLASSES,
    transformation=eval_transformation
)

test_data = FITDataset(
    image_test_path,
    mask_test_path,
    class_name=CLASSES,
    transformation=eval_transformation
)

# Hyperparameters
batch_size = 32
shuffle = True
num_workers = 0 

# Dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

