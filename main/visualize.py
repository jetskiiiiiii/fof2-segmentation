import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataset import FITDataset
from transformation import apply_transformation

# Visualize data
def visualize(image=None, mask=None):
    """PLot images in one row."""

    plt.figure(figsize=(10, 5))
    if image is not None:
        plt.subplot(1, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title("Image")
        plt.imshow(image)

    if mask is not None:
        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask)

    plt.show()


if __name__ == "__main__":
    DATA_DIRECTORY = "dataset/"
    #x_train_directory = os.path.join(, "")
    #y_train_directory = os.path.join(, "")
    #dataset = FITDataset(x_train_directory, y_train_directory)
    #image, mask = dataset[0]

    sample_image_path = os.path.join(DATA_DIRECTORY, "FTIF_LTPMP-14-Feb-2019.png")
    image = mpimg.imread(sample_image_path)

    transformation = apply_transformation()
    transformed_image = transformation(image=image)['image']

    visualize(image=transformed_image)
