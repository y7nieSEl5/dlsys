from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        if img.ndim == 1:
            img = img.reshape(28, 28)

        if self.transforms:
            # Add channel dimension -> (28, 28, 1)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            for t in self.transforms:
                img = t(img)
            return img, self.labels[index]
        else:
            # No transforms: return flattened image
            return img.reshape((-1, 28 * 28)), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as f:
        image_header = f.read(16)
        if not isinstance(image_header, (bytes, bytearray)):
            raise ValueError("Invalid image header")
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", image_header)
        if magic != 2051:
            raise ValueError("Invalid MNIST image file magic number")

        image_data = f.read(num_images * num_rows * num_cols)
        if not isinstance(image_data, (bytes, bytearray)):
            raise ValueError("Invalid image data")

        X = np.frombuffer(image_data, dtype=np.uint8).reshape(
            num_images, num_rows * num_cols
        )
        X = X.astype(np.float32) / 255.0

    with gzip.open(label_filename, "rb") as f:
        label_header = f.read(8)
        if not isinstance(label_header, (bytes, bytearray)):
            raise ValueError("Invalid label header")
        magic, num_labels = struct.unpack(">II", label_header)
        if magic != 2049:
            raise ValueError("Invalid MNIST label file magic number")

        label_data = f.read(num_labels)
        if not isinstance(label_data, (bytes, bytearray)):
            raise ValueError("Invalid label data")
        y = np.frombuffer(label_data, dtype=np.uint8)

    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of images does not match number of labels")

    return X, y
    ### END YOUR CODE