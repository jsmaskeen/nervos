"""
This module implements the `MNISTLoader` class, which extends the `Dataloader` 
base class. It is specifically designed to handle the MNIST dataset, providing 
methods for:

    - Loading and filtering digit classes
    - Normalizing images
    - Converting images into spike trains
    - Retrieving balanced or random samples

The `MNISTLoader` supports preprocessing for Spiking Neural Network (SNN) 
training workflows.
"""

from ..utils import *
from .loader import Dataloader
from sklearn.datasets import fetch_openml
from typing import Union
from scipy.fft import dct, idct
from sklearn.decomposition import PCA


class MNISTLoader(Dataloader):
    """
    A dataloader class specifically designed for loading and preprocessing the MNIST dataset.

    This class extends the `Dataloader` base class and provides functionality
    for loading MNIST data, normalizing images, converting images into spike
    trains, and retrieving balanced or random samples.

    Attributes:
        parameters (Parameters): Configuration parameters for the loader.
        classes (list): List of digit classes to include in the dataset.
        X_train (np.ndarray): Training images.
        Y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing images.
        Y_test (np.ndarray): Testing labels.
    """

    def __init__(
        self, parameters: Parameters, classes: list = [i for i in range(10)]
    ) -> None:
        """
        Initializes the MNISTLoader with given parameters and filters classes.

        Args:
            parameters (Parameters): Configuration parameters.
            classes (list, optional): List of digit classes to filter. Defaults to all 10 classes.
        """
        super().__init__(parameters)
        self.parameters = parameters
        logger.info("Loading Raw Data")
        mnist__ = fetch_openml('mnist_784', version=1)
        X__ = np.array(mnist__.data, dtype="uint8").reshape(-1, 28, 28)
        y__ = np.array(mnist__.target, dtype="int64")
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = (X__[:60000], y__[:60000]), (X__[60000:], y__[60000:])
        if len(classes) != 0:
            train_filter = np.isin(self.Y_train, classes)
            self.X_train = self.X_train[train_filter]
            self.Y_train = self.Y_train[train_filter]
            test_filter = np.isin(self.Y_test, classes)
            self.X_test = self.X_test[test_filter]
            self.Y_test = self.Y_test[test_filter]

    def normalise(self, img: np.ndarray) -> np.ndarray:
        """
        Normalizes an image to the range [0, 1].

        Args:
            img (np.ndarray): The image to normalize.

        Returns:
            np.ndarray: The normalized image.
        """
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def img2spiketrain(self, img: np.ndarray) -> np.ndarray:
        """
        Converts a normalized image into a spike train representation.

        Args:
            img (np.ndarray): The normalized image.

        Returns:
            np.ndarray: The spike train representation of the image.
        """
        sx, sy = img.shape
        time_steps = self.parameters.training_duration + 1
        normalized_img = self.normalise(img)
        frequencies = (
            normalized_img
            * (self.parameters.max_frequency - self.parameters.min_frequency)
            + self.parameters.min_frequency
        )
        intervals = np.ceil(self.parameters.training_duration / frequencies).astype(int)
        spike_trains = np.zeros((sx, sy, time_steps), dtype=int)

        for t in range(1, time_steps):
            spikes = (t % intervals == 0) & (img > 0)  # spike condition
            spike_trains[:, :, t] = spikes.astype(int)

        return spike_trains.reshape(-1, time_steps)

    def get_random_image(
        self, get_spike_train: bool = False
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Retrieves a random image and its label from the training dataset.

        Args:
            get_spike_train (bool, optional): Whether to return the spike train of the image. Defaults to False.

        Returns:
            tuple: A tuple containing the image, label, and optionally the spike train.
        """
        idx = np.random.randint(0, len(self.X_train))
        X = self.X_train[idx]
        Y = self.Y_train[idx]
        spike_train = None
        if get_spike_train:
            spike_train = self.img2spiketrain(X)
        return X, Y, spike_train

    def load_balanced_mnist(
        self, Y: np.ndarray, num_samples: int, seed: int = None
    ) -> np.ndarray:
        """
        Balances the dataset by selecting an equal number of samples for each label.

        Args:
            Y (np.ndarray): The labels of the dataset.
            num_samples (int): Total number of samples to select.
            seed (int, optional): Seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: Indices of the selected samples.
        """
        rng = np.random.default_rng(seed)
        unique_labels = np.unique(Y)
        samples_per_label = num_samples // len(unique_labels)
        selected_indices = []

        for label in unique_labels:
            indices = np.where(Y == label)[0]
            selected_indices.extend(
                rng.choice(indices, size=samples_per_label, replace=False)
            )

        selected_indices = rng.permutation(selected_indices)
        return selected_indices

    def compress_image(self, image: np.ndarray, k: int) -> np.ndarray:
        """
        Compress image by keeping top block of m x m = k coefficients.

        Args:
            image (np.ndarray): 2D numpy array (e.g. 28x28 for MNIST)
            k (int): number of coefficients to keep.
            (k must be chosen so that sqrt(k) is an integer; we keep a top-left block of size m x m, where m = sqrt(k))
        Returns:
            np.ndarray: The DCT coefficient matrix with only the top-left m x m block kept.
        """
        image_dct = dct(dct(image.T, norm="ortho").T, norm="ortho")
        m = int(np.sqrt(k))
        mask = np.zeros_like(image_dct)
        mask[:m, :m] = 1
        compressed_dct = image_dct * mask
        return compressed_dct

    def uncompress_image(
        self, image: np.ndarray, k: int, threshold: float = 0.5
    ) -> np.ndarray:
        """
        Given the (possibly sparsified) DCT coefficients, reconstruct the image using the inverse DCT.

        Args:
            image (np.ndarray): 2D numpy array.
            k (int): number of coefficients to keep.
            threshold (float): The threshold below which all pixel values will be zero after normalization. Defaults to 0.5

        Returns:
            np.ndarray: The uncompressed image.
        """
        img = idct(idct(image.T, norm="ortho").T, norm="ortho")
        uncompressed = (img - np.min(img)) / (np.max(img) - np.min(img))
        uncompressed[uncompressed < threshold] = 0
        uncompressed[uncompressed >= threshold] = 1
        uncompressed = (uncompressed - np.min(uncompressed)) / (
            np.max(uncompressed) - np.min(uncompressed)
        )
        m = int(np.sqrt(k))
        return uncompressed[:m, :m]

    def image2dct2image(
        self, image: np.ndarray, k: int, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compresses the image, then keeps top k DCT features, then uncompresses the image.

        Args:
            image (np.ndarray): 2D numpy array (e.g. 28x28 for MNIST)
            k (int): number of coefficients to keep.
            (k must be chosen so that sqrt(k) is an integer; we keep a top-left block of size m x m, where m = sqrt(k))
            threshold (float): The threshold below which all pixel values will be zero after normalization. Defaults to 0.5

        Returns:
            np.ndarray: The DCT coefficient matrix with only the top-left m x m = k block kept.
            np.ndarray: The top k DCT features uncompressed image.

        """
        img = self.compress_image(image.astype(np.float32), k)
        return img, self.uncompress_image(img,k, threshold)
    
    
    def dataloader(
        self,
        train: bool = True,
        preprocess: bool = False,
        random_single: bool = False,
        seed: int = 42,
        size: int = None,
        k: int = None,
        threshold: float = 0.5,
        pca=False
    ) -> tuple[np.ndarray, Union[np.ndarray, int]]:
        """
        Loads and preprocesses the MNIST dataset.

        Args:
            train (bool, optional): Whether to load training or testing data. Defaults to True.
            preprocess (bool, optional): Whether to preprocess the data into spike trains. Defaults to False.
            random_single (bool, optional): Whether to return a single random sample. Defaults to False.
            seed (int, optional): Seed for reproducibility. Defaults to 42.
            size (int, optional): Number of samples to load. Defaults to None.
            k (int, optional): The top k features to keep after taking DCT of the image. Defaults to None means take the whole image.
            threshold (float): Used only if DCT is used. The threshold below which all pixel values will be zero after normalization. Defaults to 0.5
            pca (bool): If compression should happen according to PCA.

        Returns:
            np.ndarray or tuple: Preprocessed data and labels, or a single random sample.
        """
        if pca:
            pPCA = PCA(n_components=0.95, svd_solver='full')
        if not random_single:
            fin_X = []
            fin_Y = []

            logger.info(
                f"Loading{' preprocessed' if preprocess else ''} {'train' if train else 'test'} data"
            )
            if train:
                if not size:
                    size = self.parameters.training_images_amount
                indices = self.load_balanced_mnist(self.Y_train, size, seed=seed)
                X, Y = self.X_train[indices], self.Y_train[indices]
            else:
                if not size:
                    size = self.parameters.testing_images_amount
                indices = self.load_balanced_mnist(self.Y_test, size, seed=seed)
                X, Y = self.X_test[indices], self.Y_test[indices]

            if pca==True:
                n_samples, h, w = X.shape
                X_flat = X.reshape((n_samples, h * w))
                X = pPCA.fit_transform(X_flat)
                print(X.shape)
                
            for img, label in zip(X, Y):
                if isinstance(k, int) and k > 0:
                    _, img = self.image2dct2image(img, k, threshold)
                if preprocess:
                    if pca:
                        img = img.reshape((1,-1))
                    img = self.img2spiketrain(img)
                fin_X.append(img)
                fin_Y.append(label)

            return np.array(fin_X), np.array(fin_Y)
        idx = np.random.choice(range(len(self.X_train)))
        img = self.X_train[idx]
        if isinstance(k, int) and k > 0:
            _, img = self.image2dct2image(img, k, threshold)
        elif pca:
            img = pPCA.fit_transform(img.reshape((1,-1)))[0]
        if preprocess:
            if pca:
                img = img.reshape((1,-1))
            return (
                img,
                self.img2spiketrain(img),
                self.Y_train[idx],
            )

        return img, img, self.Y_train[idx]
