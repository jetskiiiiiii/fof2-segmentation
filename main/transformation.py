import albumentations as A

original_x_min, original_x_max, original_y_min, original_y_max = 100, 692, 50, 540
original_height = original_y_max - original_y_min
original_width = original_x_max - original_x_min
TARGET_DIMS = 640

"""
Transformation inspired by https://albumentations.ai/docs/examples/example-kaggle-salt/
"""
train_transformation = A.Compose([
    A.Crop(x_min=original_x_min, x_max=original_x_max, y_min=original_y_min, y_max=original_y_max), # Based on trial and error
    A.PadIfNeeded(min_height=TARGET_DIMS, min_width=TARGET_DIMS, p=1),

    A.D4(p=1),

    # Not needed if applying D4
   # A.OneOf(
   #     [
   #         A.HorizontalFlip(p=0.5),
   #         A.VerticalFlip(p=0.5),
   #         A.RandomRotate90(p=0.5),
   #         A.Transpose(p=0.5),
   #     ],
   #     p=0.9,
   # ),


    A.OneOf(
        [
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, p=0.5),
            A.RandomSizedCrop(min_max_height=(100, 400), size=(original_height, original_width), p=0.5),
        ],
        p=0.9,
    ),

    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),
    A.RandomGamma(p=0.8),
    A.Resize(height=TARGET_DIMS, width=TARGET_DIMS),
])

eval_transformation = A.Compose([
    A.Crop(x_min=original_x_min, x_max=original_x_max, y_min=original_y_min, y_max=original_y_max), # Based on trial and error
    A.PadIfNeeded(min_height=TARGET_DIMS, min_width=TARGET_DIMS, p=1),
    A.Resize(height=TARGET_DIMS, width=TARGET_DIMS),
])
