import matplotlib.pyplot as plt
from PIL import Image

from data.dataset import *


def mnist_png_dataset_to_numpy(train_path, val_path, numpy_path):
    """
    Extracts the necessary files from the DatasetURLs.MNIST link.
    """

    if numpy_path.exists():
        return

    numpy_path.mkdir()

    x = []
    y = []

    for digit in train_path.iterdir():
        for image in digit.iterdir():
            x.append(np.array(Image.open(image)))
            y.append(digit.name)

    for digit in val_path.iterdir():
        for image in digit.iterdir():
            x.append(np.array(Image.open(image)))
            y.append(digit.name)

    np.save(numpy_path / "x", x)
    np.save(numpy_path / "y", y)


def show_examples_images(x, y):
    """
    Plots a 5x5 grid of sample images.
    """

    col = 4
    row = 4
    fig = plt.figure(figsize=(10, 10))

    for i in range(1, col * row + 1):
        rand_idx = np.random.randint(10)
        img = x[rand_idx]
        label = y[rand_idx]

        fig.add_subplot(row, col, i)
        plt.imshow(img)
        plt.title(label)

        plt.xticks([])
        plt.yticks([])

    plt.show()


if __name__ == "__main__":

    # Download data if necessary
    mnist_url = DatasetURLs.MNIST
    download_path = pathlib.Path.home() / "Desktop" / "MNIST"
    download_data(mnist_url, download_path)

    # Loading images into numpy arrays
    train_path = download_path / "mnist_png" / "training"
    val_path = download_path / "mnist_png" / "testing"
    numpy_path = download_path / "numpy"

    mnist_png_dataset_to_numpy(train_path, val_path, numpy_path)

    # Create dataset
    x = np.load(numpy_path / "x.npy")
    y = np.load(numpy_path / "y.npy")

    dataset = Dataset(
        x=x,
        y=y,
    )

    # Some visualization
    show_examples_images(dataset.x_train, dataset.y_train)
