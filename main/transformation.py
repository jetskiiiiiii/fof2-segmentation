import cv2 as cv
import albumentations as A
import matplotlib.pyplot as plt

original_x_min, original_x_max, original_y_min, original_y_max = 100, 697, 50, 582
#original_x_min, original_x_max, original_y_min, original_y_max = 100, 692, 50, 540
original_height = original_y_max - original_y_min
original_width = original_x_max - original_x_min
TARGET_DIMS = 640

"""
Transformation inspired by https://albumentations.ai/docs/examples/example-kaggle-salt/
"""
train_transformation = A.Compose([
    A.Crop(x_min=original_x_min, x_max=original_x_max, y_min=original_y_min, y_max=original_y_max), # Based on trial and error
    #A.PadIfNeeded(min_height=TARGET_DIMS, min_width=TARGET_DIMS, p=1),

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
    #A.PadIfNeeded(min_height=TARGET_DIMS, min_width=TARGET_DIMS, p=1),
    A.Resize(height=TARGET_DIMS, width=TARGET_DIMS),
])

if __name__ == "__main__":
    path_to_image = f"./dataset/test/test_images/FTIF_LTPMP-1-Apr-2019.jpg"
    image = cv.imread(path_to_image) 
    assert image is not None, "File could not be read."
    image = eval_transformation(image=image)
    image = image["image"]
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # For compatibility with Matplotlib

    dpi = 100
    fig_dim = 640 / dpi

    fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=dpi)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.imshow(
        image,
        zorder=0
    )

    ax.axis("off")
    plt.savefig("./predictions/for_testing/x.jpg", format='jpg', pad_inches=0)
    plt.close()
