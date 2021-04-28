"""
File defines the Dataset class and provides quick links to popular datasets for training.
"""

import math
import pathlib
import tarfile
import urllib.request

import numpy as np


def download_data(url, local_path):
    """
    Downloads data from url and places files at local_path.
    """

    download_path = pathlib.Path(local_path)

    if download_path.exists():
        return
    else:
        response = urllib.request.urlopen(url)
        tar = tarfile.open(fileobj=response, mode="r|gz")
        tar.extractall(download_path)


class DatasetURLs:
    """
    Provides links to prebuilt datasets.

    See https://course.fast.ai/datasets for more datasets.
    """

    MNIST = "https://s3.amazonaws.com/fast-ai-imageclas/mnist_png.tgz"
    CIFAR_10 = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    PETS = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz"
    FLOWERS = "https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz"
    FOOD = "https://s3.amazonaws.com/fast-ai-imageclas/food-101.tgz"


class Dataset:
    """
    Builds a dataset for model training.
    """

    def __init__(self, x, y, split=(0.7, 0.2, 0.1), shuffle=True, normalize=True):
        """
        Assumes the shape of x: [num_items, x1, x2, ..., xN] and y: [num_items, y1, y2, ..., yM].

        Split assumes the decimal values in the following order: (training, validation, testing).
        """

        self.x_raw = np.array(x)
        self.y_raw = np.array(y)
        self.num_items = self.x_raw.shape[0]
        self.split = split

        # Shuffles samples along the num_items axis
        if shuffle:
            shuffler = np.random.permutation(self.x_raw.shape[0])

            self.x_raw = self.x_raw[shuffler]
            self.y_raw = self.y_raw[shuffler]

        # Normalizes the data
        if normalize:
            mean = np.mean(self.x_raw)
            std = np.std(self.x_raw)

            self.x_raw = (self.x_raw - mean) / std

        # Perform split of data into training, validation, and testing sets
        if not math.isclose(sum(self.split), 1):
            raise ValueError("Split of dataset must sum to 1")

        num_train = math.ceil(self.split[0] * self.num_items)
        
        # If no testing set
        if self.split[2] == 0:
            num_val = self.num_items - num_train
            num_test = 0
        else:
            num_val = math.ceil(self.split[1] * self.num_items)
            num_test = self.num_items - num_train - num_val

        train_idx_stop = num_train
        val_idx_stop = num_train + num_val
        test_idx_stop = num_train + num_val + num_test

        self.x_train, self.y_train = self.x_raw[:train_idx_stop], self.y_raw[:train_idx_stop]
        self.x_val, self.y_val = self.x_raw[train_idx_stop:val_idx_stop], self.y_raw[train_idx_stop:val_idx_stop]
        self.x_test, self.y_test = self.x_raw[val_idx_stop:test_idx_stop], self.y_raw[val_idx_stop:test_idx_stop]
