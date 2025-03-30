"""
This module implements the `IrisLoader` class, which extends the `Dataloader` 
base class. It is specifically designed to handle the Iris dataset, providing 
methods for:
- Loading the dataset
- Generating more synthetic data in the dataset
- Converting samples into spike trains

The `IrisLoader` supports preprocessing for Spiking Neural Network (SNN) 
training workflows.
"""

from ..utils import *
from .loader import Dataloader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class IrisLoader(Dataloader):
    def __init__(self, parameters: Parameters, num_rf: tuple, var: tuple) -> None:
        """
        Initializes the IrisLoader with the given parameters and loads the Iris dataset.

        Args:
            parameters (Parameters): Configuration parameters for the dataloader.
            num_rf (tuple): Number of receptive fields for each feature dimension.
            var (tuple): Variance values for Gaussian encoding.
        """
        super().__init__(parameters)
        self.parameters = parameters
        self.X, self.y = load_iris(return_X_y=True)
        self.var = var
        self.num_rf = num_rf

    def fluff_data(self, samples_per_class: int) -> None:
        """
        Generates additional synthetic data by adding noise to the existing samples.

        Args:
            samples_per_class (int): Number of synthetic samples to generate per class.
        """
        new_X = list(self.X)
        new_Y = list(self.y)
        classes = np.unique(self.y)
        for _cls in classes:
            X_class = self.X[self.y == _cls]
            mu = X_class.mean(axis=0)
            std = X_class.std(axis=0)

            for _ in range(samples_per_class):
                new_X.append(
                    mu + 1.5 * np.random.rand() * np.random.randn(*mu.shape) * std
                )
                new_Y.append(_cls)
        self.X = np.array(new_X)
        self.y = np.array(new_Y)

    def generate_data(self, total_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a new dataset with balanced classes using Gaussian-distributed noise.

        Args:
            total_samples (int): Total number of samples to generate.

        Returns:
            tuple[np.ndarray, np.ndarray]: Generated feature matrix and corresponding labels.
        """
        classes = np.unique(self.y)
        samples_per_class = total_samples // len(classes)

        new_X = []
        new_Y = []

        for _cls in classes:
            X_class = self.X[self.y == _cls]
            mu = X_class.mean(axis=0)
            std = X_class.std(axis=0)

            class_samples = mu + np.random.randn(samples_per_class, *mu.shape) * std
            new_X.append(class_samples)
            new_Y.extend([_cls] * samples_per_class)

        return np.vstack(new_X), np.array(new_Y)

    def normalise(self) -> None:
        """
        Normalizes the dataset features to the range [0, 1].
        """
        arr_min = self.X.min(axis=0)
        arr_max = self.X.max(axis=0)
        self.X = (self.X - arr_min) / (arr_max - arr_min)

    def encode_coordinates(self, point: np.ndarray) -> np.ndarray:
        """
        Encodes a given feature vector using Gaussian receptive fields.

        Args:
            point (np.ndarray): Feature vector to encode.

        Returns:
            np.ndarray: Encoded representation using Gaussian functions.
        """
        axes = [np.linspace(0, 1, self.num_rf[i]) for i in range(len(self.num_rf))]
        gaussians = [
            np.exp(-((point[i] - axes[i]) ** 2) / (2 * self.var[i]))
            for i in range(len(self.num_rf))
        ]
        return np.concatenate(gaussians)

    def generate_poisson_spikes(
        self, spike_probs: np.ndarray, time_steps: int
    ) -> np.ndarray:
        """
        Converts encoded feature vectors into Poisson-distributed spike trains.

        Args:
            spike_probs (np.ndarray): Probability values for spike generation.
            time_steps (int): Number of time steps for the spike train.

        Returns:
            np.ndarray: Spike train representation of the input.
        """
        normalized_probs = spike_probs / np.max(spike_probs)

        firing_frequencies = self.parameters.min_frequency + normalized_probs * (
            self.parameters.max_frequency - self.parameters.min_frequency
        )

        spike_trains = np.zeros((len(spike_probs), time_steps), dtype=int)
        intervals = np.ceil(
            (self.parameters.training_duration + 1) / firing_frequencies
        ).astype(int)

        for t in range(1, time_steps):
            spikes = t % intervals == 0
            spike_trains[:, t] = spikes.astype(int)

        return spike_trains

    def dataloader(
        self,
        size: int = None,
        train: bool = True,
        preprocess: bool = False,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads and optionally preprocesses the Iris dataset for SNN training.

        Args:
            size (int, optional): Number of samples to load (for test set only).
            train (bool, optional): Whether to load training or test data. Defaults to True.
            preprocess (bool, optional): Whether to convert features into spike trains. Defaults to False.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            tuple[np.ndarray, np.ndarray]: Feature matrix and labels.
        """
        fin_X = []
        self.normalise()
        tr_X, te_X, tr_y, te_y = train_test_split(
            self.X,
            self.y,
            test_size=(self.parameters.testing_images_amount)
            / (
                self.parameters.testing_images_amount
                + self.parameters.training_images_amount
            ),
            random_state=seed,
        )
        if train:
            X = tr_X
            y = tr_y
        else:
            X = te_X
            y = te_y

        if size:
            if train:
                raise "DOnt pls dont"
            X, y = self.generate_data(size)
        for point in X:
            if preprocess:
                point = self.generate_poisson_spikes(
                    self.encode_coordinates(point),
                    self.parameters.training_duration + 1,
                )
            fin_X.append(point)
        return np.array(fin_X), y
