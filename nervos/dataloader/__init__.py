"""
This package provides classes and methods for loading, preprocessing, and 
managing datasets, with a focus on Spiking Neural Network (SNN) applications.

Submodules:
    mnist: MNIST-specific dataloader with SNN preprocessing support.
    loader: Base dataloader class for custom datasets.
    circles: Non-linear circle dataset loader with SNN preprocessing support.
    iris: Iris dataset loader with SNN preprocessing support.
"""

from .mnist import *
from .loader import *
from .circles import *
from .iris import *