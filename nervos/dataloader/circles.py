"""
This module implements the `CirclesLoader` class, which extends the `Dataloader` 
base class. It is specifically designed to handle scikit-learn's circle toy dataset, and 
provides methods for:

    - Loading data
    - Converting points into spike trains using Gaussian receptive fields
    - Retrieving train and test data

The `CirclesLoader` supports preprocessing for Spiking Neural Network (SNN) 
training workflows.
"""


from ..utils import *
from .loader import Dataloader
from sklearn.datasets import make_circles


class CirclesLoader(Dataloader):
    """
    A dataloader class specifically designed for loading and preprocessing the make_circles dataset.

    This class provides functionality for:
    
        - Loading the dataset
        - Normalizing coordinates
        - Encoding coordinates into Gaussian probabilities
        - Generating spike trains based on Poisson processes
        - A dataloader method for retrieving data samples

    Attributes:
        parameters (Parameters): Configuration parameters for the loader.
        noise (float): Standard deviation of Gaussian noise added to the data.
        factor (float): Scale factor between the inner and outer circle.
    """

    def __init__(
        self,
        parameters: Parameters,
        num_rf: tuple[int, int],
        var: tuple[float, float],
        noise: float = 0.06,
        factor: float = 0.5,
    ) -> None:
        """
        Initializes the CirclesLoader with given parameters.

        Args:
            parameters (Parameters): Configuration parameters for spike generation.
            num_rf (tuple[int, int]): Number of receptive fields for the x and y axes.
            var (tuple[float, float]): Variance of the Gaussian receptive fields along the x and y axes.
            noise (float, optional): Standard deviation of Gaussian noise added to the data. Defaults to 0.06.
            factor (float, optional): Scale factor between the inner and outer circle. Defaults to 0.5.
        """
        self.noise = noise
        self.factor = factor
        self.parameters = parameters
        self.num_rf = num_rf
        self.var = var

    def encode_coordinates(self, point: np.ndarray) -> np.ndarray:
        """
        Encodes a coordinate into Gaussian probabilities for each neuron.

        Args:
            point (np.ndarray): A 2D point (x, y) to encode.

        Returns:
            np.ndarray: Encoded probabilities for each neuron.
        """
        point+=1
        axes_x = np.linspace(0, 2, self.num_rf[0])
        axes_y = np.linspace(0, 2, self.num_rf[1])
        gaussians_x = np.exp(-((point[0] - axes_x) ** 2) / (2 * self.var[0]))
        gaussians_y = np.exp(-((point[1] - axes_y) ** 2) / (2 * self.var[1]))

        return np.concatenate([gaussians_x, gaussians_y])

    def generate_poisson_spikes(
        self, spike_probs: np.ndarray, time_steps: int
    ) -> np.ndarray:
        """
        Generates Poisson spike trains for a given probability distribution.

        Args:
            spike_probs (np.ndarray): Probabilities for each neuron to fire.
            time_steps (int): Number of time steps for the spike train.

        Returns:
            np.ndarray: Spike trains for all neurons.
        """
        normalized_probs = spike_probs / np.max(spike_probs)

        firing_frequencies = self.parameters.min_frequency + normalized_probs * (
            self.parameters.max_frequency - self.parameters.min_frequency
        )

        spike_trains = np.zeros((len(spike_probs), time_steps), dtype=int)
        intervals = np.ceil((self.parameters.training_duration+1) / firing_frequencies).astype(
            int
        )

        for t in range(1, time_steps):
            spikes = t % intervals == 0
            spike_trains[:, t] = spikes.astype(int)

        return spike_trains

    def dataloader(
        self,
        size: int=None,
        train: bool = True,
        preprocess: bool = False,
        seed: int = 42,
    )->tuple[np.ndarray,np.ndarray]:
        """
        Loads and preprocesses the dataset.

        Args:
            train (bool, optional): Placeholder for train/test split. Defaults to True.
            size (int, optional): Number of samples to load. Defaults to None (all samples).
            preprocess (bool, optional): Whether to preprocess the data into spike trains. Defaults to False.
            seed (int, optional): Seed for reproducibility. Defaults to 42.

        Returns:
            tuple[np.ndarray, np.ndarray]: Preprocessed data and labels.
        """
        fin_X = []
        logger.info(
            f"Loading{' preprocessed' if preprocess else ''} {'train' if train else 'test'} data"
        )
        if train:
            if not size:
                size = self.parameters.training_images_amount
            data,labels = make_circles(size,noise=self.noise,factor=self.factor,random_state=seed)
        else:
            if not size:
                size = self.parameters.testing_images_amount
            data,labels = make_circles(size,noise=self.noise,factor=self.factor,random_state=seed+10)
        
        for point in data:
            if preprocess:
                point = self.generate_poisson_spikes(
                self.encode_coordinates(point), self.parameters.training_duration+1
            )
            fin_X.append(point)
        return np.array(fin_X),labels
    
    
    def plot_rf(self) -> None:
        """
        Plots the receptive fields and covered area.
        """
        data,labels = make_circles(400,noise=self.noise,factor=self.factor)
        data+=1
        plt.figure(figsize=(17,7))
        
        axes_x = np.linspace(0, 2, self.num_rf[0])
        axes_y = np.linspace(0, 2, self.num_rf[1])
        vals = np.arange(-0.7,2.7,0.01)
        plt.subplot(1,2,1)
        plt.scatter(data[:,0][labels==0],data[:,1][labels==0],alpha=0.5,c='orange')
        plt.scatter(data[:,0][labels==1],data[:,1][labels==1],alpha=0.5,c='blue')
        for idx,mean in enumerate(axes_x):
            gaussians_x = 2*np.exp(-((vals - mean) ** 2) / (2 * self.var[0]))
            plt.plot(vals,gaussians_x,label=f"Neuron {idx+1}")
            plt.fill_between(vals,gaussians_x,alpha=0.2)
            plt.grid(True,'both')
            plt.legend(loc="upper right")
        plt.xticks(axes_x)
        plt.yticks([])
            
        plt.subplot(1,2,2)
        plt.scatter(data[:,0][labels==0],data[:,1][labels==0],alpha=0.5,c='orange')
        plt.scatter(data[:,0][labels==1],data[:,1][labels==1],alpha=0.5,c='blue')
        for idx,mean in enumerate(axes_y):
            gaussians_y = 2*np.exp(-((vals - mean) ** 2) / (2 * self.var[1]))
            
            plt.plot(gaussians_y,vals,label=f"Neuron {idx+1+self.num_rf[0]}")
            plt.fill_between(gaussians_y,vals,alpha=0.2)
            plt.grid(True,'both')
            plt.legend(loc="upper right")
        plt.xticks([])
        plt.yticks(axes_y)
        
