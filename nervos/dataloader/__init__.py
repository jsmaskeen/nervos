"""
This package provides classes and methods for loading, preprocessing, and 
managing datasets, with a focus on Spiking Neural Network (SNN) applications. 

Modules:
- loader: Base dataloader class for custom datasets.
- mnist_loader: MNIST-specific dataloader with SNN preprocessing support.
- circles_loader: Non linear circles-specific dataloader with SNN preprocessing support.
- iris_loader: Iris-specific dataloader with SNN preprocessing support.
"""

from .mnist import *
from .loader import *
from .circles import *
from .iris import *